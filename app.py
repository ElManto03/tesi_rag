from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import os
import logging
import shutil
import tempfile
import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from jose import jwt
from db_manager import create_db_and_tables, ottenere_utente_da_db # Importa la funzione di creazione tabelle
from file_processor import process_single_file # Importa la funzione di elaborazione file
from chatbot import get_query_embedding, retrieve_context, generate_answer, is_query_safe, ask_question
from file_server import router as file_router
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    create_db_and_tables()
    yield

JWT_SECRET_KEY = os.getenv("JWT_ENCRYPTION_KEY")
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

ALGORITHM = "HS256"

app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY)
app.include_router(file_router)
oauth = OAuth()

oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

@app.middleware("http")
async def blindaggio_autenticazione(request: Request, call_next):
    # 1. Definiamo le uniche rotte "pubbliche" che Google o il sistema devono poter raggiungere
    rotte_pubbliche = [
        "/login",          # La rotta che avvia il bridge con Google
        "/auth/callback",  # La rotta di ritorno da Google
    ]
    
    # Se l'utente sta cercando di andare al login o al callback, lo lasciamo passare liberamente
    if request.url.path in rotte_pubbliche:
        return await call_next(request)
    
    # 2. Controlliamo se esiste il token nei cookie di sessione
    token_sessione = request.cookies.get("session_token")
    
    # 3. SE IL TOKEN NON C'È: Blocchiamo l'accesso e deviamo l'utente sulla rotta /login
    if not token_sessione:
        # Questo forza il browser a caricare immediatamente il flusso Google OAuth
        return RedirectResponse(url="/login")
    
    # SE IL TOKEN C'È: L'utente è autenticato, procediamo normalmente verso la chat o le API
    response = await call_next(request)
    return response

@app.get('/login')
async def login(request: Request):
    # Reindirizza l'utente alla pagina di login di Google
    redirect_uri = request.url_for('auth_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get('/auth/callback')
async def auth_callback(request: Request):
    # 1. Recupera i dati da Google
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get('userinfo')
    email_utente = user_info.get('email')
    
    if not email_utente:
        raise HTTPException(status_code=400, detail="Impossibile recuperare l'email da Google.")

    # 2. INTERROGA IL TUO DATABASE POSTGRES
    # (Adatta questa riga alla logica con cui interroghi il DB, es. SQLAlchemy o connessione diretta)
    utente_db = ottenere_utente_da_db(email_utente) 
    
    # Se l'email non è censita nel DB (quindi non è un prof/responsabile autorizzato)
    if not utente_db:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accesso negato. Questo account non è autorizzato ad accedere al sistema scolastico."
        )
        
    # 3. GENERA IL TUO JWT LOCALE (I dati che serviranno alle Row Level Security)
    scadenza = datetime.datetime.utcnow() + datetime.timedelta(hours=8) # La sessione dura 8 ore
    payload_jwt = {
        "sub": email_utente,
        "role": utente_db.user_role,    # es. 'teacher'
        "exp": scadenza
    }
    
    token_locale = jwt.encode(payload_jwt, JWT_SECRET_KEY, algorithm=ALGORITHM)
    
    # 4. CREA IL REINDIRIZZAMENTO VERSO LA CHAT (es. la rotta Radice "/")
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    
    # 5. SALVA IL TOKEN NEI COOKIE DEL BROWSER
    response.set_cookie(
        key="session_token",
        value=token_locale,
        httponly=True,   # CRITICO PER LA SICUREZZA: impedisce a script JS malevoli di rubare il token (Mitiga XSS)
        max_age=28800,   # Durata in secondi (8 ore)
        samesite="lax"   # Protegge da attacchi CSRF
    )
    
    return response

@app.get("/", response_class=HTMLResponse)
async def main_interface():
    """
    Interfaccia web con menu per navigare tra Chatbot e Upload.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Test App</title>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; background-color: #f0f2f5; }
            nav { background: #001529; color: white; padding: 1rem; display: flex; gap: 20px; }
            nav a { color: #a6adb4; text-decoration: none; cursor: pointer; font-weight: 500; }
            .role-selector { margin-left: auto; align-self: center; font-size: 0.9rem; }
            .role-selector select { padding: 4px; border-radius: 4px; border: none; background: #002140; color: white; }
            .level-selector { align-self: center; font-size: 0.9rem; margin-left: 10px; color: white; }
            .level-selector select { padding: 4px; border-radius: 4px; border: none; background: #002140; color: white; }
            .language-selector { align-self: center; font-size: 0.9rem; margin-left: 10px; color: white; }
            .language-selector select { padding: 4px; border-radius: 4px; border: none; background: #002140; color: white; }
            nav a.active { color: white; border-bottom: 2px solid #1890ff; }
            .container { padding: 20px; flex-grow: 1; }
            .section { display: none; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            .section.active { display: block; }
            .chat-box { height: 300px; border: 1px solid #d9d9d9; margin-bottom: 10px; padding: 10px; overflow-y: auto; background: #fafafa; }
            input[type="text"], select { padding: 8px; border: 1px solid #d9d9d9; border-radius: 4px; }
            button { padding: 8px 16px; background: #1890ff; color: white; border: none; border-radius: 4px; cursor: pointer; margin: 5px 0; }
            button:hover { background: #40a9ff; }
            button#stop-btn:hover { background: #ff7875; }
            button#stop-btn { display: none; }
            button#stop-btn.visible { display: inline-block; }
            .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
            .message p { margin: 5px 0; }
            .message ul, .message ol { padding-left: 25px; margin: 5px 0; }
            .message h1, .message h2, .message h3 { font-size: 1.2rem; margin: 10px 0 5px 0; }
            .message code { background: #eee; padding: 2px 4px; border-radius: 3px; font-family: monospace; }
            .user-msg { background: #e6f7ff; border-left: 4px solid #1890ff; }
            .bot-msg { background: #f6ffed; border-left: 4px solid #52c41a; }
            .sources { font-size: 0.8rem; color: #888; margin-top: 5px; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #f8f9fa; }
            .status-waiting { color: #888; }
            .status-working { color: #1890ff; font-weight: bold; }
            .status-done { color: #52c41a; font-weight: bold; }
            .status-error { color: #f5222d; font-weight: bold; }
            #drop-zone {
                border: 2px dashed #1890ff;
                border-radius: 8px;
                padding: 40px;
                text-align: center;
                color: #1890ff;
                background-color: #e6f7ff;
                cursor: pointer;
                margin-top: 15px;
                transition: background-color 0.3s;
            }
            #drop-zone.dragover { background-color: #bae7ff; border-color: #40a9ff; }
        </style>
    </head>
    <body>
        <nav>
            <a id="nav-chat" onclick="showSection('chat')" class="active">Chatbot</a>
            <a id="nav-upload" onclick="showSection('upload')">Upload Documenti</a>
            <div class="role-selector">
                Ruolo:
                <select id="sim-role" onchange="updateRole()">
                    <option value="developer">Developer (Admin)</option>
                    <option value="user">User (Sola Lettura)</option>
                </select>
            </div>
            <div class="level-selector">
                Livello:
                <select id="sim-level" onchange="updateRole()">
                    <option value="public">Pubblico</option>
                    <option value="private">Privato</option>
                </select>
            </div>
            <div class="language-selector">
                Lingua:
                <select id="sim-lang" onchange="updateRole()">
                    <option value="italiano">Italiano</option>
                    <option value="inglese">Inglese</option>
                    <option value="francese">Francese</option>
                    <option value="tedesco">Tedesco</option>
                    <option value="spagnolo">Spagnolo</option>
                </select>
            </div>
        </nav>

        <div class="container">
            <!-- Sezione Chatbot -->
            <div id="section-chat" class="section active">
                <h2>Interroga il Chatbot</h2>
                <div class="chat-box" id="chat-output">Benvenuto! Carica un documento o fammi una domanda...</div>
                <input type="text" id="chat-input" style="width: 55%;" placeholder="Scrivi qui la tua domanda...">
                <button id="send-btn" onclick="sendMessage()">Invia</button>
                <button id="stop-btn" onclick="stopMessage()" style="background-color: #ff4d4f;">Ferma</button>
                <button id="block-btn" onclick="blockPrompt()" style="background-color: #fa8c16;">Blocca</button>
            </div>

            <!-- Sezione Upload -->
            <div id="section-upload" class="section">
                <h2>Carica Nuovo Documento</h2>
                <p>Seleziona file o una cartella per caricarli massivamente. Sono accettati solo file .docx, .html e .pdf (consigliato).</p>
                <label for="fileInput"><b>Carica cartella:</b></label>
                <input type="file" id="fileInput" multiple webkitdirectory mozdirectory>
                <p><b>O carica file singoli qui sotto:</b></p>

                <input type="file" id="hiddenFileInput" accept=".pdf,.docx,.html" multiple style="display: none;">
                <div id="drop-zone">Trascina qui i tuoi file .pdf, .docx o .html (o clicca per selezionarli)</div>

                <div id="upload-controls" style="display:none; margin-top: 20px;">
                    <button onclick="processAll()">Avvia Elaborazione Massiva</button>
                    <table>
                        <thead>
                            <tr>
                                <th>Nome File</th>
                                <th>Access Level</th>
                                <th>OCR Forzato (consigliato per moduli)</th>
                                <th>Scadenza (Opzionale)</th>
                                <th>Stato</th>
                                <th>Azioni</th>
                            </tr>
                        </thead>
                        <tbody id="file-table-body"></tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            let currentAbortController = null;

            marked.use({
                renderer: {
                    link(token) {
                        // Estraiamo le proprietà dall'oggetto token
                        const href = token.href || '';
                        const title = token.title ? `title="${token.title}"` : '';
                        const text = token.text || href;
                        
                        return `<a href="${href}" ${title} target="_blank" rel="noopener noreferrer">${text}</a>`;
                    }
                }
            });
            
            function updateRole() {
                const role = document.getElementById('sim-role').value;
                const level = document.getElementById('sim-level').value;
                const lang = document.getElementById('sim-lang').value;
                localStorage.setItem('simRole', role);
                localStorage.setItem('simLevel', level);
                localStorage.setItem('simLang', lang);
                // Se cambiamo in user mentre siamo in upload, torniamo alla chat
                const currentSection = localStorage.getItem('activeSection') || 'chat';
                showSection(role === 'user' ? 'chat' : currentSection);
            }

            function showSection(id) {
                const role = localStorage.getItem('simRole') || 'developer';
                const level = localStorage.getItem('simLevel') || 'public';
                const lang = localStorage.getItem('simLang') || 'italiano';
                document.getElementById('sim-role').value = role;
                document.getElementById('sim-level').value = level;
                document.getElementById('sim-lang').value = lang;

                // UI Restriction: L'utente vede solo il chatbot
                document.getElementById('nav-upload').style.display = (role === 'user') ? 'none' : 'inline';
                if (role === 'user' && id === 'upload') id = 'chat';

                document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
                document.querySelectorAll('nav a').forEach(a => a.classList.remove('active'));
                document.getElementById('section-' + id).classList.add('active');
                document.getElementById('nav-' + id).classList.add('active');
                localStorage.setItem('activeSection', id);
            }

            // Al caricamento della pagina, ripristina l'ultima sezione visitata
            document.addEventListener('DOMContentLoaded', () => {
                updateRole();
            });

            let selectedFiles = [];

            // Gestione selezione file (sia cartella che file multipli)
            const handleFileSelection = (e) => handleFiles(e.target.files);

            function handleFiles(fileList) {
                const files = Array.from(fileList).filter(f =>
                    f.name.endsWith('.pdf') || f.name.endsWith('.docx') || f.name.endsWith('.html')
                );

                if (files.length === 0) {
                    alert("Nessun file .pdf, .docx o .html valido trovato nella selezione. Si prega di selezionare file supportati.");
                    document.getElementById('upload-controls').style.display = 'none';
                    return;
                }

                if (files.length > 0) {
                    selectedFiles = [...selectedFiles, ...files];
                    document.getElementById('upload-controls').style.display = 'block';
                    renderTable();
                }
            }

            // Drag and Drop Logic
            const dropZone = document.getElementById('drop-zone');
            const hiddenFileInput = document.getElementById('hiddenFileInput'); // Nuovo input nascosto
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(name => {
                dropZone.addEventListener(name, (e) => { e.preventDefault(); e.stopPropagation(); });
            });
            dropZone.addEventListener('dragover', () => dropZone.classList.add('dragover'));
            dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
            dropZone.addEventListener('drop', (e) => {
                dropZone.classList.remove('dragover');
                handleFiles(e.dataTransfer.files);
            });
            dropZone.addEventListener('click', () => hiddenFileInput.click()); // Attiva il click sull'input nascosto

            document.getElementById('fileInput').addEventListener('change', handleFileSelection);
            hiddenFileInput.addEventListener('change', handleFileSelection); // Ascolta il cambio del nuovo input nascosto

            function renderTable() {
                const tbody = document.getElementById('file-table-body');
                tbody.innerHTML = '';

                selectedFiles.forEach((file, index) => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${file.name}</td>
                        <td>
                            <select id="access-${index}">
                                <option value="public">Pubblico</option>
                                <option value="private">Privato</option>
                            </select>
                        </td>
                        <td>
                            <select id="ocr-${index}">
                                <option value="no">No</option>
                                <option value="sì">Sì</option>
                            </select>
                        </td>
                        <td>
                            <input type="date" id="expiry-${index}">
                        </td>
                        <td id="status-${index}" class="status-waiting">In attesa...</td>
                        <td>
                            <button onclick="removeFile(${index})" style="background-color: #ff4d4f; padding: 5px 10px;">Rimuovi</button>
                        </td>
                    `;
                    tbody.appendChild(tr);
                });
            }

            function removeFile(index) {
                selectedFiles.splice(index, 1);
                if (selectedFiles.length === 0) {
                    document.getElementById('upload-controls').style.display = 'none';
                }
                renderTable();
            }

            async function processAll() {
                for (let i = 0; i < selectedFiles.length; i++) {
                    const statusCell = document.getElementById(`status-${i}`);
                    if (statusCell.innerText === "Completato") continue;

                    statusCell.innerText = "In elaborazione...";
                    statusCell.className = "status-working";

                    const role = localStorage.getItem('simRole') || 'developer';
                    const formData = new FormData();
                    formData.append('file', selectedFiles[i]);
                    formData.append('access_level', document.getElementById(`access-${i}`).value);
                    formData.append('module_option', document.getElementById(`ocr-${i}`).value);
                    formData.append('expiry_date', document.getElementById(`expiry-${i}`).value);
                    formData.append('db_role', role);

                    try {
                        const response = await fetch('/uploadfile/', { method: 'POST', body: formData });
                        const result = await response.json();

                        if (response.ok) {
                            statusCell.innerText = "Completato";
                            statusCell.className = "status-done";
                        } else {
                            statusCell.innerText = "Errore: " + result.message;
                            statusCell.className = "status-error";
                        }
                    } catch (e) {
                        statusCell.innerText = "Errore di rete";
                        statusCell.className = "status-error";
                    }
                }
            }

            function blockPrompt() {
                if (currentAbortController) {
                    currentAbortController.abort();
                    console.log("[DEBUG RAG] Pulsante Stop cliccato. Richiesta annullata.");
                } else {
                    console.log("[DEBUG RAG] Nessuna generazione attiva da interrompere.");
                }
            }

            async function sendMessage() {
                const input = document.getElementById('chat-input');
                const text = input.value.trim();
                if (!text) return;

                const chatOutput = document.getElementById('chat-output');
                const role = localStorage.getItem('simRole') || 'developer';
                const level = localStorage.getItem('simLevel') || 'public';
                const lang = localStorage.getItem('simLang') || 'italiano';

                // Aggiungi messaggio utente
                chatOutput.innerHTML += `<div class="message user-msg"><b>Tu:</b> ${text}</div>`;
                input.value = '';
                chatOutput.scrollTop = chatOutput.scrollHeight;

                // ─── NUOVA LOGICA ABORT CONTROLLER ───
                // Se c'era una generazione precedente ancora attiva, la interrompiamo per sicurezza
                if (currentAbortController) {
                    currentAbortController.abort();
                }
                // Creiamo un nuovo controller per questa specifica richiesta
                currentAbortController = new AbortController();
                const signal = currentAbortController.signal;
                // ─────────────────────────────────────

                try {
                    const response = await fetch('/chat/', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: text, role: role, level: level, language: lang }),
                        signal: signal
                    });

                    if (!response.ok) {
                        const data = await response.json();
                        let errorMsg = "Si è verificato un errore.";
                        if (data && data.detail) {
                            errorMsg = data.detail.message || (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
                        }
                        chatOutput.innerHTML += `<div class="message bot-msg" style="color:red"><b>Errore di Sicurezza:</b> ${errorMsg}</div>`;
                        chatOutput.scrollTop = chatOutput.scrollHeight;
                        return; // Blocca l'esecuzione qui
                    }

                    // 2. GESTIONE STREAM: Accumuliamo tutto in memoria prima di mostrare
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder("utf-8");
                    let fullText = "";

                    // Il server sta generando e controllando... noi accumuliamo e basta
                    while (true) {
                        const { value, done } = await reader.read();
                        if (done) break; 

                        const token = decoder.decode(value, { stream: true });
                        fullText += token;
                    }

                    // ─── QUI LO STREAM È FINITO E IL CONTROLLO RELEVANCE È PASSATO ───

                    // 3. CONTROLLO SE IL BACKEND HA RILEVATO UN BLOCCO DOPO LA GENERAZIONE
                    if (fullText.startsWith("[SECURITY_BLOCKED]")) {
                        const cleanMsg = fullText.replace("[SECURITY_BLOCKED]", "").trim();
                        chatOutput.innerHTML += `
                            <div class="message bot-msg security-warning" style="border-left: 4px solid #ff9800; background-color: #fff3e0; padding: 8px;">
                                <b>Sistema di Sicurezza (Output):</b> ${marked.parse(cleanMsg)}
                            </div>`;
                    } else {
                        // 4. CASO STANDARD: Tutto sicuro, mostriamo l'intera risposta d'un colpo
                        chatOutput.innerHTML += `<div class="message bot-msg"><b>Bot:</b> ${marked.parse(fullText)}</div>`;
                    }

                    
                } catch (e) {
                    // 🪲 CONTROLLO SE L'ERRORE È DOVUTO ALL'INTERRUZIONE VOLONTARIA
                    if (e.name === 'AbortError') {
                        console.log("[DEBUG RAG] Generazione interrotta dall'utente.");
                        chatOutput.innerHTML += `
                            <div class="message bot-msg" style="color:#fa8c16; background-color: #fffb8f; border-left: 4px solid #fa8c16; padding: 8px;">
                                <i>Generazione interrotta dall'utente.</i>
                            </div>`;
                    } else {
                        // Gestione dei normali errori di rete
                        console.error("[DEBUG RAG] Errore di rete:", e);
                        chatOutput.innerHTML += `
                            <div class="message bot-msg" style="color:red; background-color: #fff1f0; border-left: 4px solid #ff4d4f; padding: 8px;">
                                <b>Errore di Rete:</b> Impossibile connettersi al server.
                                <br><small style="color: #555;">Dettaglio: ${e.message}</small>
                            </div>`;
                    }
                } finally {
                    // Una volta terminata la richiesta (con successo o errore), azzeriamo il controller
                    currentAbortController = null;
                }
                chatOutput.scrollTop = chatOutput.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/chat/")
async def chat_endpoint(payload: dict, request: Request):
    question = payload.get("question")
    role = payload.get("role", "user")
    level = payload.get("level", "public")
    language = payload.get("language", "italiano")

    if not question:
        raise HTTPException(status_code=400, detail="Domanda mancante")

    return StreamingResponse(
        ask_question(
            query=question,
            db_role=role,
            user_level=level,
            language=language,
            request=request
        ),
        media_type="text/event-stream"
    )

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...), access_level: str = Form(...), module_option: str = Form(...), expiry_date: str = Form(None), db_role: str = Form("user")):
    """
    Endpoint per caricare un file, processarlo e salvare gli embedding nel DB.
    """
    # Protezione lato Server: Blocca l'azione se il ruolo simulato non è developer
    if db_role != "developer":
        raise HTTPException(status_code=403, detail="Azione di caricamento non consentita per il ruolo Utente.")

    if not file.filename:
        return {"message": "Nessun file selezionato"}

    # Crea una directory temporanea per salvare il file
    with tempfile.TemporaryDirectory() as tmpdir:
        file_location = os.path.join(tmpdir, file.filename)
        try:
            # Salva il file temporaneamente
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Prepara i metadati richiesti da process_single_file
            metadata = {
                "access_level": access_level,
                "module_option": module_option,
                "expiry_date": expiry_date
            }

            # Processa il file usando la logica di ocr_to_md.py
            chunks = process_single_file(file_location, output_dir="output_dir", metadata=metadata, db_role=db_role)

            if chunks:
                return {"message": f"File '{file.filename}' caricato e processato con successo. Generati {len(chunks)} chunk."}
            else:
                return {"message": f"File '{file.filename}' caricato, ma nessun chunk generato o errore nel processo."}
        except Exception as e:
            print(f"Errore durante l'elaborazione del file: {e}")
            return {"message": f"Errore durante l'elaborazione del file: {e}"}, 500
