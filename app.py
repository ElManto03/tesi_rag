from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import os
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from sqlmodel import Session, create_engine

from config import settings
from ocr_to_md import process_single_file # Importa la funzione di elaborazione file


engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI))


def create_db_and_tables() -> None:
    # Nota: Assicurati che le tabelle 'documents' e 'document_chunks' siano create 
    # o tramite SQLModel qui o tramite migrazioni/script SQL esterni.
    pass


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    create_db_and_tables()
    yield


app = FastAPI(lifespan=lifespan)


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
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; background-color: #f0f2f5; }
            nav { background: #001529; color: white; padding: 1rem; display: flex; gap: 20px; }
            nav a { color: #a6adb4; text-decoration: none; cursor: pointer; font-weight: 500; }
            nav a.active { color: white; border-bottom: 2px solid #1890ff; }
            .container { padding: 20px; flex-grow: 1; }
            .section { display: none; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            .section.active { display: block; }
            .chat-box { height: 300px; border: 1px solid #d9d9d9; margin-bottom: 10px; padding: 10px; overflow-y: auto; background: #fafafa; }
            input[type="text"], select { padding: 8px; border: 1px solid #d9d9d9; border-radius: 4px; }
            button { padding: 8px 16px; background: #1890ff; color: white; border: none; border-radius: 4px; cursor: pointer; margin: 5px 0; }
            button:hover { background: #40a9ff; }
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
        </nav>

        <div class="container">
            <!-- Sezione Chatbot -->
            <div id="section-chat" class="section active">
                <h2>Interroga il Chatbot</h2>
                <div class="chat-box" id="chat-output">Benvenuto! Carica un documento o fammi una domanda...</div>
                <input type="text" id="chat-input" style="width: 70%;" placeholder="Scrivi qui la tua domanda...">
                <button onclick="sendMessage()">Invia</button>
            </div>

            <!-- Sezione Upload -->
            <div id="section-upload" class="section">
                <h2>Carica Nuovo Documento</h2>
                <p>Seleziona file o una cartella per caricarli massivamente. Sono accettati solo file .docx e .pdf (consigliato).</p>
                <label for="fileInput"><b>Carica cartella:</b></label>
                <input type="file" id="fileInput" multiple webkitdirectory mozdirectory>
                <p><b>O carica file singoli qui sotto:</b></p>

                <input type="file" id="hiddenFileInput" accept=".pdf,.docx" multiple style="display: none;">
                <div id="drop-zone">Trascina qui i tuoi file .pdf o .docx (o clicca per selezionarli)</div>

                <div id="upload-controls" style="display:none; margin-top: 20px;">
                    <button onclick="processAll()">Avvia Elaborazione Massiva</button>
                    <table>
                        <thead>
                            <tr>
                                <th>Nome File</th>
                                <th>Access Level</th>
                                <th>OCR Forzato (consigliato per moduli)</th>
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
            function showSection(id) {
                document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
                document.querySelectorAll('nav a').forEach(a => a.classList.remove('active'));
                document.getElementById('section-' + id).classList.add('active');
                document.getElementById('nav-' + id).classList.add('active');
                localStorage.setItem('activeSection', id);
            }

            // Al caricamento della pagina, ripristina l'ultima sezione visitata
            document.addEventListener('DOMContentLoaded', () => {
                const activeSection = localStorage.getItem('activeSection') || 'chat';
                showSection(activeSection);
            });

            let selectedFiles = [];

            // Gestione selezione file (sia cartella che file multipli)
            const handleFileSelection = (e) => handleFiles(e.target.files);

            function handleFiles(fileList) {
                const files = Array.from(fileList).filter(f =>
                    f.name.endsWith('.pdf') || f.name.endsWith('.docx')
                );
                
                if (files.length === 0) {
                    alert("Nessun file .pdf o .docx valido trovato nella selezione. Si prega di selezionare file supportati.");
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

                    const formData = new FormData();
                    formData.append('file', selectedFiles[i]);
                    formData.append('access_level', document.getElementById(`access-${i}`).value);
                    formData.append('module_option', document.getElementById(`ocr-${i}`).value);

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

            function sendMessage() {
                alert("Funzionalità chatbot non ancora collegata all'endpoint di ricerca!");
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...), access_level: str = Form(...), module_option: str = Form(...)):
    """
    Endpoint per caricare un file, processarlo e salvare gli embedding nel DB.
    """
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
                "module_option": module_option
            }

            # Processa il file usando la logica di ocr_to_md.py
            chunks = process_single_file(file_location, output_dir="output_dir", metadata=metadata)

            if chunks:
                return {"message": f"File '{file.filename}' caricato e processato con successo. Generati {len(chunks)} chunk."}
            else:
                return {"message": f"File '{file.filename}' caricato, ma nessun chunk generato o errore nel processo."}
        except Exception as e:
            print(f"Errore durante l'elaborazione del file: {e}")
            return {"message": f"Errore durante l'elaborazione del file: {e}"}, 500
