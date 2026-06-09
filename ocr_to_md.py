import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import base64
import requests
import subprocess
import torch
import shutil
import pypandoc
import json
import re
import pysbd
import io
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader
from pdf2image import convert_from_path, pdfinfo_from_path
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from llama_index.core.node_parser import SemanticSplitterNodeParser, MarkdownNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Document, Settings
from config import settings
from PIL import ImageEnhance, Image
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
from docling.document_converter import DocumentConverter
import ollama
#from paddleocr import PaddleOCRVL

# Configurazione cartella debug per immagini OCR
DEBUG_DIR = "ocr_debug_images"
# if os.path.exists(DEBUG_DIR):
#     shutil.rmtree(DEBUG_DIR)
# os.makedirs(DEBUG_DIR, exist_ok=True)

MAX_PAGES_TABLE_OCR = 4

def custom_sentence_splitter(text):
    return LegalSegmenter().split_sentences(text)

def clean_junk_text(text):
    """Rimuove le righe di boilerplate della scuola dal testo markdown."""
    junk_patterns = [
        r'^#*\s*ISTITUTO TECNICO TECNOLOGICO STATALE.*$',
        r'^#*\s*"ODONE BELLUZZI - LEONARDO DA VINCI".*$',
        r'^#*\s*RIMINI.*$',
        r'^#*\s*Via Ada Negri, 34 - 47923 Rimini.*$',
        r'^#*\s*Tel\..*$',
        r'^#*\s*Web: ittsrimini\.edu\.it.*$',
        r'^#*\s*PEC: RNTF010004@pec\.istruzione\.it.*$',
    ]
    cleaned_lines = []
    for line in text.splitlines():
        if line.strip() and any(re.match(p, line.strip(), re.IGNORECASE) for p in junk_patterns):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

class LegalSegmenter:
    def __init__(self):
        self.seg = pysbd.Segmenter(language="it", clean=False)
        # Acronimi comuni nella PA e scuola italiana per evitare split errati
        self.abbreviations = [
            'a.s', 'd.p.r', 'd.lgs', 'p.t.o.f', 'd.m', 'n', 'art', 'comma', 'lett', 'all', 'p.e.i', 'p.d.p'
        ]

    def split_sentences(self, text):
        # Protezione acronimi: sostituisce temporaneamente il punto con un marker
        for abbr in self.abbreviations:
            text = re.sub(rf'\b({abbr})\.', r'\1__DOT__', text, flags=re.IGNORECASE)

        sentences = self.seg.segment(text)
        refined = []
        MAX_JOIN_LENGTH = 1000 # Evitiamo di unire blocchi se diventano troppo grandi

        for s in sentences:
            s = s.replace("__DOT__", ".") # Ripristina il punto reale
            
            should_join = False
            if refined:
                last_s = refined[-1].strip()
                # Unisce se la riga precedente finisce con ":" oppure è un titolo/riferimento in **grassetto**
                if (last_s.endswith(":") or (last_s.startswith("**") and last_s.endswith("**"))) \
                   and (len(last_s) + len(s) < MAX_JOIN_LENGTH):
                    should_join = True
            
            if should_join:
                refined[-1] = refined[-1] + " " + s
            else:
                refined.append(s)
        return refined

PATH_TO_MARK = r"c:\Users\federico.mantoni\AppData\Local\miniconda3\envs\tesi2\Scripts\marker_single.exe"
os.environ.setdefault('PYPANDOC_PANDOC', 'C:/Users/federico.mantoni/AppData/Local/miniconda3/envs/tesi2/Library/bin/pandoc.exe')
os.environ["LLM_SERVICE"] = "openai" 
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
os.environ["OPENAI_API_KEY"] = "ollama" # Ollama non richiede chiave, ma Marker sì
os.environ["GEN_MODEL"] = "llama3.1:8b"
scuola_info = {"indirizzo": "Via Ada Negri, 34 - 47923 Rimini (RN)", "tel": "(+39) 0541 384159",  "cf": "82007870403", "web": "itsrimini.edu.it", "mail": "RNTF010004@istruzione.it", "pec": "RNTF010004@pec.istruzione.it"}

def chunk_splitter(doc, splitter, md_parser):
    initial_nodes = md_parser.get_nodes_from_documents([doc])
    nodes = []
    
    if len(initial_nodes) > 0:
    # Soglia minima di caratteri per un chunk (es. un titolo + un paragrafo breve)
    # Sotto questa soglia, il chunk viene unito al successivo.
        MIN_CHUNK_SIZE = 200
        # Soglia massima prima di attivare lo split semantico
        MAX_CHUNK_SIZE = 2000

        buffer_text = ""
        buffer_metadata = {}

        for node in initial_nodes:
            current_text = node.text.strip()

            # Se abbiamo testo nel buffer (da un titolo corto precedente), lo uniamo
            if buffer_text:
                current_text = buffer_text + "\n\n" + current_text
                # Fondiamo i metadati (mantenendo quelli del nodo corrente come primari)
                combined_metadata = {**buffer_metadata, **node.metadata}
                buffer_text = ""
            else:
                combined_metadata = node.metadata

            # Identifichiamo se il blocco contiene una tabella Markdown
            is_table = "|" in current_text and "-|-" in current_text

            # Se il chunk risultante è ancora troppo piccolo, lo mettiamo nel buffer e passiamo oltre
            if len(current_text) < MIN_CHUNK_SIZE and not is_table:
                buffer_text = current_text
                buffer_metadata = combined_metadata
            
            else:
            # Se il chunk è di dimensioni accettabili, verifichiamo se passarlo al SemanticSplitter
                new_doc = Document(text=current_text, metadata=combined_metadata)
                if not is_table and len(current_text) > MAX_CHUNK_SIZE:
                # Se troppo grande, lo splitter semantico farà un lavoro più fine
                    splitted_nodes =splitter.get_nodes_from_documents([new_doc])
                    if any (len(node.text) < MIN_CHUNK_SIZE for node in splitted_nodes) :
                        nodes.append(new_doc)
                    else:
                        nodes.extend(splitted_nodes)
                else:
                    nodes.append(new_doc)


        # Se è rimasto qualcosa nel buffer alla fine del documento, lo appendiamo all'ultimo nodo
        if buffer_text and nodes:
            nodes[-1] = Document(text=nodes[-1].text + "\n\n" + buffer_text, metadata={**nodes[-1].metadata, **buffer_metadata})

    else:
        nodes = initial_nodes

    if nodes: print(f"Primo chunk generato: {nodes[0].get_content()[:100]}...")
    return nodes

def trova_punto_taglio_ottimale(image_path):
    """
    Trova la riga di pixel dell'immagine che contiene più del 40% di pixel neri 
    più vicina alla metà della pagina. Assume l'immagine in B/N (0=nero).
    """
    img = Image.open(image_path).convert('L')
    width, height = img.size
    meta_altezza = height // 2
    arr = np.array(img)
    
    # Poiché l'immagine è già in B/N, il nero ha valore 0.
    # Calcoliamo la percentuale di pixel neri (valore 0) per ogni riga.
    percentuale_nero = np.mean(arr < 100, axis=1)
    
    # Filtriamo le righe che hanno una densità di nero superiore al 40% SOLO sotto la metà
    indici_sotto_meta = np.where((percentuale_nero > 0.40) & (np.arange(height) > meta_altezza))[0]
    
    if len(indici_sotto_meta) > 0:
        # Scegliamo la riga più vicina alla metà (il valore minimo tra gli indici che sono già > meta_altezza)
        punto_taglio = np.min(indici_sotto_meta)
    else:
        # Fallback: se nessuna riga supera il 40%, usiamo quella con la densità massima rilevata
        punto_taglio = np.argmax(percentuale_nero)
        
    return int(punto_taglio)

def pipeline_taglio_intelligente(image_path, altezza_soglia_critica=1440):
    """
    Decide se tagliare l'immagine e, in caso positivo, esegue il taglio
    in un punto sicuro senza tranciare il testo.
    """
    img = Image.open(image_path)
    width, height = img.size
    
    # CRITERIO DEL SE: Tagliamo solo se supera la soglia critica per la VRAM
    if height <= altezza_soglia_critica:
        print(f"Immagine sicura (Altezza: {height}px). Nessun taglio necessario.")
        return [image_path]
    
    print(f"Immagine troppo grande ({height}px). Ricerca del punto di taglio sicuro...")
    
    # CRITERIO DEL DORE: Trova la riga di testo vuota
    riga_taglio = trova_punto_taglio_ottimale(image_path)
    print(f"Punto di taglio sicuro individuato alla riga: {riga_taglio}")
    
    # Eseguiamo il crop anatomico
    meta_top = img.crop((0, 0, width, riga_taglio))
    meta_bottom = img.crop((0, riga_taglio, width, height))
    
    path_top = "output_pagina_top.jpg"
    path_bottom = "output_pagina_bottom.jpg"
    
    meta_top.save(path_top, "JPEG", quality=95)
    meta_bottom.save(path_bottom, "JPEG", quality=95)
    
    return [path_top, path_bottom]

def encode_image(img_or_path):
    """Ottimizza e codifica un oggetto PIL Image in base64."""
    if isinstance(img_or_path, str):
        img = Image.open(img_or_path)
    else:
        img = img_or_path

    target_width = 1000
    if img.width > target_width:
        w_percent = (target_width / float(img.width))
        h_size = int((float(img.height) * float(w_percent)))
        img = img.resize((target_width, h_size), Image.Resampling.LANCZOS)
    grayscale_img = img.convert('L')
    enhancer = ImageEnhance.Contrast(grayscale_img)
    optimized_img = enhancer.enhance(1.4)
    #img_byte_arr = io.BytesIO()
    #optimized_img.save(img_byte_arr, format='JPEG', quality=95, optimize=True)
    # return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    # Assicura che la directory di destinazione esista
    os.makedirs(DEBUG_DIR, exist_ok=True)
    # Genera un percorso univoco per il salvataggio
    save_path = os.path.join(DEBUG_DIR, f"optimized_{int(time.time() * 1000)}.jpg")

    # Salva l'immagine ottimizzata
    optimized_img.save(save_path, format='JPEG', quality=95, optimize=True)

    return save_path

# --- CONFIGURAZIONE OCR OLLAMA ---
TABLE_OCR_OPTIONS = {"num_ctx": 4096,  "temperature": 0.0001, "num_predict": 4096}
PAGE_OCR_OPTIONS = {"num_ctx": 16384, "num_predict": 4096, "temperature": 0.2, "repeat_penalty": 1.3}

#Se una cella di una colonna è unita verticalmente e si applica a più punti elenco di un'altra colonna, devi sdoppiare la cella e RIPETERE il testo per ogni singola riga corrispondente.
CORRECTION_MODEL = "qwen2.5:14b-instruct-q4_K_M"

def _call_normalization_llm(chunk_text):
    """Esegue la chiamata all'LLM per la normalizzazione del testo Markdown."""
    correction_prompt = f"""
Tu sei un sistema di post-processing algoritmico per motori di Document Layout Analysis (DLA). 
Ricevi in input un frammento di testo (circa 3 pagine) in formato Markdown estratto da un documento. A causa dei cambi pagina, le tabelle potrebbero apparire spezzate o frammentate.

Il tuo obiettivo è normalizzare, ricostruire e consolidare la struttura logica del documento, restituendo un Markdown pulito e privo di rumore, ottimizzato per un sistema RAG.

Segui tassativamente queste linee guida ingegneristiche e universali:

1. IDENTIFICAZIONE E FUSIONE DELLE TABELLE SPEZZATE:
- Se una tabella si interrompe a causa di un cambio pagina e riprende poco dopo (riconoscibile dalla ripetizione delle stesse intestazioni di colonna o dalla continuazione logica delle righe), fondile in un'unica tabella Markdown.
- Elimina le intestazioni duplicate e i separatori di tabella (`|---|---|`) intermedi generati dal cambio pagina.

2. RISOLUZIONE DEI TRONCAMENTI SINTATTICI (SALDATURA DEL TESTO):
- Analizza la fine di ogni riga e l'inizio della successiva. Se una parola o una frase è chiaramente troncata a metà da un a capo, salda i due frammenti eliminando i trattini di sillabazione.

3. PROPAGAZIONE DELLE CELLE UNITE VERTICALMENTE (SPANNING):
- Se rilevi che una cella contiene dati strutturati importanti e le righe immediatamente successive presentano celle corrispondenti vuote (o contenenti solo spazi/simboli come '//'), propaga logicamente quel contesto. Puoi scegliere di duplicare il testo nelle righe successive oppure di consolidare tutte le sotto-infrazioni in un'unica macro-riga, separando i punti elenco interni esclusivamente con il tag HTML <br> per non rompere la riga della tabella Markdown.

4. ELIMINAZIONE DEL RUMORE DI PAGINA (STRIPPING):
- Riconosci e rimuovi completamente metadati ripetitivi che interrompono il flusso logico: numeri di pagina, intestazioni di pagina (header), piè di pagina (footer), indirizzi istituzionali, stringhe di protocollo e tag di annotazione delle immagini (es. ``).

5. PRESERVAZIONE DEL TESTO NON TABELLARE:
- Se il documento contiene paragrafi di testo normale, preservali esattamente nella loro posizione sequenziale originaria.

6. TRASFORMAZIONE DI ELENCHI IN TABELLE CONTINUATIVE:
- Se dopo una tabella incontri dei titoli seguiti da elenchi puntati o checkbox (□), comprendi che si tratta della continuazione logica della tabella precedente che il parser ha rotto.
- Converti questi elenchi in righe della tabella precedente. 
- Associa a ciascuna di queste nuove categorie i valori delle celle precedenti, senza lasciare celle vuote

7. RIMOZIONE DELLE INTESTAZIONI:
- All'inizio di ogni blocco di testo, se trovi un'intestazione che contiene il numero di telefono o il codice fiscale della scuola (82007870403), rimuovila completamente insieme a eventuali righe adiacenti che contengono informazioni di contatto o indirizzi.

OUTPUT RICHIESTO:
Restituisci esclusivamente l'intero documento normalizzato in Markdown. Non includere alcuna introduzione, alcuna spiegazione e nessun commento discorsivo (es. NON scrivere "Ecco la tabella strutturata:"). Inizia direttamente con il primo elemento valido del documento.

Ecco il testo grezzo da elaborare:
{chunk_text}
"""
    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": CORRECTION_MODEL,
                "messages": [{"role": "user", "content": correction_prompt}],
                "stream": False,
                "options": {"num_ctx": 32768, "temperature": 0.1}
            },
            timeout=600
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"⚠️ Errore normalizzazione LLM: {e}")
        return chunk_text

def llm_normalize_markdown(md_text, separator_pattern, pages_per_chunk=3):
    """Divide il markdown in blocchi e chiama l'LLM per la normalizzazione."""
    print(f"--- 🚀 Normalizzazione LLM ({CORRECTION_MODEL}) ---")
    md_text = re.sub(r'<!--\s*image\s*-->', '', md_text)
    matches = list(re.finditer(separator_pattern, md_text, re.MULTILINE | re.IGNORECASE))
    if not matches:
        return _call_normalization_llm(md_text)
    final_text = ""
    if matches[0].start() > 0:
        final_text += md_text[:matches[0].start()].strip() + "\n\n"
    for i in range(0, len(matches), pages_per_chunk):
        start_pos = matches[i].start()
        end_pos = matches[i + pages_per_chunk].start() if i + pages_per_chunk < len(matches) else len(md_text)
        chunk_text = md_text[start_pos:end_pos]
        if not chunk_text.strip(): continue
        print(f"🔧 Elaborazione blocco di pagine partendo da {matches[i].group(0).strip()}...")
        corrected_chunk = _call_normalization_llm(chunk_text)
        page_tag = matches[i].group(0).strip()
        if not re.match(r'\{\d+\}-+', page_tag):
            page_idx_match = re.search(r'(\d+)', page_tag)
            page_idx = page_idx_match.group(1) if page_idx_match else i
            page_tag = f"{{{page_idx}}}------------------------------------------------"
        final_text += page_tag + "\n\n" + corrected_chunk + "\n\n"
    requests.post("http://localhost:11434/api/generate", json={"model": CORRECTION_MODEL, "keep_alive": 0})
    return final_text

TABLE_OCR_PROMPT = """Tu sei un OCR visivo di altissima precisione. Il tuo compito è convertire l'immagine di questa tabella complessa in formato Markdown pulito e strutturato.

Segui queste istruzioni tassative:

1. VIETATO RIASSUMERE O PARAFRASARE: Non ottimizzare il testo, non accorciare le frasi e non cambiare le parole (es. se c'è scritto 'collegamento internet per motivi personali, non di studio e ricerca', NON devi scrivere 'internet per motivi non scolastici'). Devi copiare ogni singola parola, inclusi gli articoli e le congiunzioni, carattere per carattere. Se accorci una frase, l'output è considerato ERRORE.

2. TRASCRIZIONE DEI TITOLI E DEL CONTESTO:
Inizia trascrivendo fedelmente i titoli o i testi esplicativi presenti sopra la tabella usando la formattazione Markdown standard (## o ###).

3. REGOLE DI STRUTTURA DELLA TABELLA (Uso del <br> per elenchi):
- È tassativamente vietato l'uso di elenchi puntati all'interno della tabella, sia usando la sintassi Markdown (- o *), sia usando tag HTML (<ul>, <li>).
- Se una cella contiene più punti elenco, frasi distinte o sotto-voci, trascrivili tutti all'interno della STESSA cella, separando ogni punto esclusivamente con il tag HTML <br> per andare a capo senza rompere la riga Markdown.
- GESTIONE DELLE CELLE UNITE VERTICALMENTE: Se una macro-categoria si applica a un intero blocco di elementi, crea un'unica grande riga Markdown. Inserisci tutti gli elementi corrispondenti nella cella adiacente separati da <br>, in modo che rimangano perfettamente allineati visivamente con le rispettive colonne di riferimento.

4. REGOLE DI RIGORE PER L'OUTPUT:
- Inizia direttamente con il primo titolo o riga di testo rilevata in alto.
- NON inserire introduzioni, saluti, commenti discorsivi o spiegazioni del tipo "Ecco la tabella convertita". Restituisci ESCLUSIVAMENTE l'output in Markdown.
- Sii un OCR letterale: trascrivi ogni stringa esattamente come appare visivamente, senza riassumere, omettere o parafrasare il testo."""
PAGE_OCR_PROMPT = """Trascrivine il contenuto in Markdown standard.
Se trovi a inizio pagina una tabella con più colonne ma una sola di queste è non vuota, considerala come il continuo di una precedente tabella e uniscila ad essa.
Se vedi parti di tabella che hanno una sola riga e colonna, trasformali in un elenco puntato.
Se vedi parti di tabella con una sola colonna e una sola parola al suo interno, trasformalo in un titolo.
Assicurati di scrivere tutto il testo nelle tabelle, se non riesci a dividere correttamente una riga di una tabella, trasformala in un paragrafo normale ma mantieni il testo.
Mantieni titoli e strutture. Solo testo Markdown, no commenti."""

def _call_ollama_ocr(image, prompt, options):
    """Centralizza la chiamata a Ollama per ridurre la duplicazione e gestire la VRAM."""
    headers = {"Connection": "close"}
    try:
        print("inizio chiamata OCR Ollama...")
        
        response = ollama.chat(
            model='qwen2.5vl:3b',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image],
                'options': options
            }]
        )
        # response = requests.post(
        #     "http://localhost:11434/api/chat",
        #     json={
        #         "model": "richardyoung/olmocr2:7b-q8",
        #         "prompt": prompt,
        #         "images": [image],
        #         "stream": False,
        #         "options": options
        #     },
        #     headers=headers, timeout=300
        # )
        #response.raise_for_status()
        text = (response['message']['content']).strip()
        requests.post("http://localhost:11434/api/generate", json={"model": "qwen2.5vl:3b", "keep_alive": 0})
        return clean_junk_text(text)
    except Exception as e:
        print(f"❌ Errore OCR Ollama: {e}")
        return ""

def get_node_page_number(node):
    """
    Estrae il numero di pagina dai metadati del nodo.
    La logica di fallback sul testo è ora gestita in run_semantic_chunking.
    """
    metadata = getattr(node, "metadata", {}) or {}
    # Marker usa spesso indici 0-based, potresti voler aggiungere +1 a seconda delle esigenze
    page = metadata.get("page_number") or metadata.get("page")
    if page is not None:
        try:
            return int(page)
        except (ValueError, TypeError):
            return None
    return None

def get_total_pages(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        try:
            reader = PdfReader(file_path)
            return len(reader.pages)
        except Exception:
            return None
    elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        return 1
    return None

# Inizializziamo l'engine una sola volta a livello di modulo per gestire meglio 
# il pool di connessioni, invece di ricrearlo per ogni file salvato.
db_engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI), future=True)


def save_document_and_chunks_to_db(file_name, file_url, access_level, total_pages, chunks):
    if not access_level:
        raise ValueError("Access level mancante; rifiuto il caricamento del documento.")

    try:
        with db_engine.begin() as conn:
            document_id = conn.execute(
                text(
                    "INSERT INTO documents (file_name, file_url, access_level, total_pages) "
                    "VALUES (:file_name, :file_url, :access_level, :total_pages) RETURNING id"
                ),
                {
                    "file_name": file_name,
                    "file_url": file_url,
                    "access_level": access_level,
                    "total_pages": total_pages,
                },
            ).scalar_one()

            chunk_rows = []
            for chunk in chunks:
                chunk_rows.append(
                    {
                        "parent_doc_id": document_id,
                        "content": chunk["text"],
                        "chunk_index": chunk["chunk_id"],
                        "page_number": chunk.get("page_number"),
                        "embedding": chunk.get("embedding"),
                    }
                )

            if chunk_rows:
                conn.execute(
                    text(
                        "INSERT INTO document_chunks "
                        "(parent_doc_id, content, chunk_index, page_number, embedding) "
                        "VALUES (:parent_doc_id, :content, :chunk_index, :page_number, :embedding)"
                    ),
                    chunk_rows,
                )

        print(f"✅ Documento salvato su DB con id {document_id} e {len(chunk_rows)} chunk.")
        return document_id
    except SQLAlchemyError as exc:
        print(f"❌ Errore DB: {exc}")
        raise


def run_semantic_chunking(text,  metadata=None, model_name="qwen3-embedding:8b"):
    print(f"--- 🧩 Avvio Semantic Chunking con LlamaIndex ({model_name}) ---")
    print("metadata:", metadata)

    text = clean_junk_text(text)

    embed_model = OllamaEmbedding(
        model_name=model_name,
        base_url="http://localhost:11434",
    )

    md_parser = MarkdownNodeParser()

    splitter = SemanticSplitterNodeParser(
        buffer_size=3, # Aumentato leggermente per dare più contesto alle finestre di analisi
        breakpoint_percentile_threshold=95, # Più sensibile ai cambi di argomento
        embed_model=embed_model,
        sentence_splitter=custom_sentence_splitter
    )

    if not metadata or "access_level" not in metadata or not metadata.get("access_level"):
        raise ValueError("Access level mancante: non posso procedere con l'embedding del documento.")

    combined_metadata = {**scuola_info}
    combined_metadata.update(metadata)

    doc = Document(
        text=text,
        metadata=combined_metadata,
        excluded_embed_metadata_keys=["cf", "mail", "pec", "indirizzo", "tel", "web", "source_path", "access_level"]
    )

    nodes = chunk_splitter(doc, splitter, md_parser)
    print(f"✅ Generati {len(nodes)} chunk semantici.")

    # --- GESTIONE PAGINE E METADATI ---
    # Poiché la prima pagina spesso non ha tag e i chunk sono sequenziali,
    # manteniamo lo stato della pagina corrente durante l'iterazione.
    current_page = 1 
    for node in nodes:
        content = node.get_content()
        # Cerca il tag {N}------------------ nel contenuto del chunk
        page_tags = re.findall(r'\{(\d+)\}-+', content)
        if page_tags:
            # Il separatore di Marker è 0-based, prendiamo l'ultima occorrenza trovata nel chunk
            current_page = int(page_tags[-1]) + 1
        
        # Iniettiamo il metadato direttamente nel nodo LlamaIndex
        node.metadata["page_number"] = current_page

        # Rimuove il separatore dal testo del chunk per pulizia (come richiesto)
        cleaned_content = re.sub(r'\{(\d+)\}-+', '', content).strip()
        node.set_content(cleaned_content)

    # --- BATCH EMBEDDING ---
    chunk_texts = [node.get_content() for node in nodes]
    embeddings = embed_model.get_text_embedding_batch(chunk_texts)

    return [
        {
            "chunk_id": i,
            "text": node.get_content(),
            "page_number": get_node_page_number(node),
            "embedding": embedding,
        }
        for i, (node, embedding) in enumerate(zip(nodes, embeddings))
    ]

def convert_docx_to_md(docx_path, output_dir="output_dir"):
    print(f"--- Conversione Word nativa: {docx_path} ---")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(docx_path))[0]
    output_path = os.path.join(output_dir, base_name + ".md")

    # Converte docx in markdown preservando le tabelle
    try:
        format_with_extensions = (
            "markdown"
            "-bracketed_spans"
            "-native_spans"
            "-inline_code_attributes"
            "-header_attributes"
        )

        pypandoc.convert_file(
            docx_path,
            format_with_extensions,
            format='docx',
            extra_args=['--wrap=none'],
            outputfile=output_path
        )

        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    except Exception as e:
        print(f"❌ Errore durante la conversione di {docx_path}: {e}")
        return None


def run_marker_pdf(pdf_path, output_dir):
    pdf_path = os.path.abspath(pdf_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    base_output_dir = os.path.join(output_dir, base_name)
    nested_md = os.path.join(base_output_dir, base_name + ".md")

    if not os.path.exists(pdf_path):
        print(f"Errore: Il file PDF non esiste al percorso: {pdf_path}")
        return None

    print(f"--- Tentativo con Marker-PDF: {pdf_path} ---")
    
    try:
        # Usiamo shell=True su Windows per trovare l'eseguibile nel PATH
        subprocess.run([
            PATH_TO_MARK,
            pdf_path,
            "--output_dir", output_dir,
            "--disable_ocr",
            "--paginate_output", # Disabilitiamo l'OCR interno di Marker per affidarlo completamente a GLM-OCR in caso di fallback
            "--debug"
        ], check=True)
        
        # Pulisci immagini estratte con peso < 30 KB che iniziano con "_"
        if os.path.exists(base_output_dir):
            for file_name in os.listdir(base_output_dir):
                if file_name.startswith("_") and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    file_path_to_check = os.path.join(base_output_dir, file_name)
                    if os.path.isfile(file_path_to_check):
                        file_size_kb = os.path.getsize(file_path_to_check) / 1024
                        if file_size_kb < 30:
                            try:
                                os.remove(file_path_to_check)
                                print(f"🗑️ Eliminato: {file_name} ({file_size_kb:.2f} KB)")
                            except Exception as e:
                                print(f"⚠️ Errore eliminazione {file_name}: {e}")
        
        # ... resto del codice per leggere il file ...
        if os.path.exists(nested_md):
            print("✅ Marker ha generato il file markdown")
            with open(nested_md, "r", encoding="utf-8") as f:
                return f.read()
    except subprocess.CalledProcessError as e:
        print(f"Marker ha restituito un errore: {e}")
    except FileNotFoundError:
        print("Errore: file non trovato.")
    return None

def run_docling_pdf(pdf_path, output_dir):
    """Esegue l'elaborazione del PDF utilizzando Docling per una conversione accurata in Markdown."""
    pdf_path = os.path.abspath(pdf_path)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    base_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(base_output_dir, exist_ok=True)
    output_path = os.path.join(base_output_dir, base_name + ".md")

    print(f"--- 🚀 Avvio Docling Document Converter: {pdf_path} ---")
    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        md_text = result.document.export_to_markdown()
        
        # Utilizziamo la funzione refactorizzata per normalizzare l'output di Docling
        # Usiamo il CF della scuola come separatore di pagina proxy per Docling
        final_corrected_text = llm_normalize_markdown(md_text, r'.*82007870403.*', pages_per_chunk=3)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_corrected_text)

        print(f"✅ Docling (e correzione a blocchi) completata: {output_path}")
        return final_corrected_text
    except Exception as e:
        print(f"❌ Errore durante l'elaborazione con Docling: {e}")
        return None

def glm_ocr(pdf_path, target_page_indices=None):
    print(f"--- 🚨 Avvio Vision-OCR (Qwen2.5-VL) ---")
    poppler_path = r"C:\Users\federico.mantoni\AppData\Local\miniconda3\envs\tesi2\Library\bin"
    
    target_indices_provided = target_page_indices is not None
    current_options = TABLE_OCR_OPTIONS if target_indices_provided else PAGE_OCR_OPTIONS
    
    if target_indices_provided:
        first = min(target_page_indices) + 1
        last = max(target_page_indices) + 1
        pages_list = convert_from_path(pdf_path, 180, first_page=first, last_page=last, poppler_path=poppler_path)
        page_map = {idx: pages_list[i] for i, idx in enumerate(range(first-1, last))}
    else:
        pages_list = convert_from_path(pdf_path, 180, poppler_path=poppler_path, thread_count=4)
        page_map = {i: p for i, p in enumerate(pages_list)}
        target_page_indices = sorted(list(page_map.keys()))
    
    # Logica semplice: processa ogni pagina singolarmente
    results_dict = {}

    # Processo sequenziale delle pagine
    target_pages = [(i, page_map[i]) for i in target_page_indices]
    for i, page in target_pages:
        # Crop: 18% in alto, 95% in basso (rimuove header e footer)
        w, h = page.size
        left, top, right, bottom = 0, int(h * 0.18), w, int(h * 0.95)
        page = page.crop((left, top, right, bottom))

        # Salvataggio debug dell'immagine croppata
        debug_path = os.path.join(DEBUG_DIR, f"page_{i+1}.jpg")
        page.save(debug_path, "JPEG")

        img_str = encode_image(page)
        prompt = TABLE_OCR_PROMPT if target_indices_provided else PAGE_OCR_PROMPT

        testo_pagina = _call_ollama_ocr([img_str], prompt, current_options)
        if testo_pagina:
            results_dict[i] = testo_pagina
            print(f"✅ Pagina {i+1} completata.")
            time.sleep(2)
        else:
            results_dict[i] = f"[Errore OCR Pagina {i+1}]"
    if target_indices_provided:
        return results_dict

    sorted_indices = sorted(results_dict.keys())
    full_text = ""
    for i in sorted_indices:
        full_text += f"\n\n{{{i}}}------------------------------------------------\n" + results_dict[i]
    return full_text.strip()

def ocr_single_image(image_path):
    """Esegue l'OCR su un singolo file immagine e restituisce il testo Markdown."""
    print(f"🖼️ Esecuzione OCR su immagine: {image_path}")
    img = Image.open(image_path)
    img_str = encode_image(img)
    # 1. Pipeline di taglio intelligente: decide se l'immagine va frammentata
    image_parts = pipeline_taglio_intelligente(image_path)
    
    ocr_texts = []
    for part_path in image_parts:
        # 2. Codifica e chiamata OCR per ogni parte (singolarmente come richiesto)
        img_str = encode_image(part_path)
        text = _call_ollama_ocr(img_str, TABLE_OCR_PROMPT, TABLE_OCR_OPTIONS)
        if text:
            ocr_texts.append(text)
        
        # 3. Pulizia file temporanei se sono stati generati dal crop
        if part_path != image_path:
            try: os.remove(part_path)
            except: pass
            
    return "\n\n".join(ocr_texts).strip()
    #return _call_ollama_ocr(img_str, TABLE_OCR_PROMPT, TABLE_OCR_OPTIONS)

def get_table_data(base_output_dir):
    """Analizza blocks.json di Marker per trovare pagine con tabelle."""
    blocks_path = os.path.join(base_output_dir, "blocks.json")
    if not os.path.exists(blocks_path):
        return set(), set()
    
    table_pages = set()
    list_group_pages = set()
    page_top_level_blocks = {} # page_id -> primi blocchi della pagina
    section_headers = set() # page_id dove è presente un titolo tabella

    def find_data_recursive(blocks):
        for block in blocks:
            b_type = str(block.get("block_type", "")).lower()
            b_desc = str(block.get("block_description", "")).lower()
            pid = block.get("page_id")
            b_text = str(block.get("text", ""))
            
            # Rilevamento Header/Titoli per definire i confini dei blocchi OCR
            # Il titolo deve contenere la parola "tab" (es. Tabella, Tab.) per essere considerato un punto di interruzione
            if ("section header" in b_desc or "title" in b_desc or b_type == "2") and ("TAB." in b_text or "TABELLA " in b_text or "CAPO " in b_text) and pid is not None:
                section_headers.add(pid)

            # Se è un blocco pagina (tipo 8), salviamo i suoi figli diretti per l'euristica
            if b_type == "8" and pid is not None:
                page_top_level_blocks[pid] = block.get("children", [])

            # Monitoriamo tipi 27 (Table) e 24 (TOC) che spesso contengono i dati strutturati
            if "table" in b_desc or "table" in b_type or b_type in ["27", "24"]:
                if pid is not None:
                    table_pages.add(pid)
            
            # Monitoriamo i List Group (tipo 6) per l'euristica di continuità
            if b_type == "6" or "list group" in b_desc:
                if pid is not None:
                    list_group_pages.add(pid)
            
            # Esplora ricorsivamente i figli se presenti (chiave 'children' in Marker)
            children = block.get("children")
            if isinstance(children, list):
                find_data_recursive(children)

    try:
        with open(blocks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Gestisce sia liste dirette che dizionari contenenti la chiave 'blocks'
            blocks_to_process = data if isinstance(data, list) else data.get("blocks", [])
            find_data_recursive(blocks_to_process)
            
        # APPLICAZIONE EURISTICA: List Group (p.x) -> Tabella nei primi 8 blocchi (p.x+1)
        sorted_pids = sorted(list(page_top_level_blocks.keys()))
        for pid in sorted_pids:
            next_pid = pid + 1
            if pid in list_group_pages and next_pid in page_top_level_blocks:
                # Controlla se nei primi 8 elementi della pagina successiva c'è una tabella
                has_early_table = False
                for b in page_top_level_blocks[next_pid][:8]:
                    bt = str(b.get("block_type", "")).lower()
                    bd = str(b.get("block_description", "")).lower()
                    if "table" in bd or "table" in bt or bt in ["27", "24"]:
                        has_early_table = True
                        break
                
                if has_early_table:
                    print(f"🔗 Euristica: Pagina {pid+1} (List Group) collegata a Tabella in Pagina {next_pid+1}")
                    table_pages.add(pid)

    except Exception as e:
        print(f"⚠️ Errore lettura blocks.json: {e}")
    return table_pages, section_headers

def assemble_hybrid_markdown(marker_md, ocr_results):
    """
    Sostituisce le pagine testuali di Marker con l'output Vision OCR.
    ocr_results: { page_idx: testo_ocr }
    """
    page_pattern = r'(\{\d+\}-+)'
    parts = re.split(page_pattern, marker_md)
    
    assembled = [parts[0]]
    for i in range(1, len(parts), 2):
        tag = parts[i]
        content = parts[i+1] if (i+1) < len(parts) else ""
        page_match = re.search(r'\{(\d+)\}', tag)
        if not page_match: continue
        current_page_idx = int(page_match.group(1))
        if current_page_idx in ocr_results:
            print(f"🔄 Ibridazione: Sostituzione pagina {current_page_idx + 1} con output Vision OCR.")
            assembled.append(tag + "\n" + ocr_results[current_page_idx])
        else:
            assembled.append(tag + content)
    return "".join(assembled)

def run_marker_pdf_with_fallback(pdf_path, output_dir, force_ocr=False):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    base_output_dir = os.path.join(os.path.abspath(output_dir), base_name)
    
    if force_ocr:
        print("--- 🚨 OCR forzata: Avvio GLM-OCR ---")
        result_text = glm_ocr(pdf_path)
        # Normalizzazione LLM
        result_text = llm_normalize_markdown(result_text, r'\{\d+\}-+', pages_per_chunk=3)
        os.makedirs(base_output_dir, exist_ok=True)
        with open(os.path.join(base_output_dir, base_name + ".md"), "w", encoding="utf-8") as f:
            f.write(result_text)
        return result_text

    # Torniamo a Marker-PDF per estrazione testo e rilevamento tabelle (tramite blocks.json)
    marker_text = run_marker_pdf(pdf_path, output_dir)

    if not marker_text or len(marker_text) < 100:
        print("⚠️ Marker ha fallito o prodotto troppo poco testo. Fallback Vision-OCR totale...")
        content = glm_ocr(pdf_path)
        os.makedirs(base_output_dir, exist_ok=True)
        with open(os.path.join(base_output_dir, base_name + ".md"), "w", encoding="utf-8") as f:
            f.write(content)
        return content

    # Logica Ibrida: cerca tabelle e usa OCR mirato per quelle pagine
    table_pages, _ = get_table_data(base_output_dir)
    if table_pages:
        print(f"📊 Rilevate tabelle nelle pagine: {sorted(list(table_pages))}. Avvio OCR pagina per pagina...")
        ocr_results = glm_ocr(pdf_path, target_page_indices=table_pages)
        marker_text = assemble_hybrid_markdown(marker_text, ocr_results)
        # Normalizzazione finale per il flusso Marker/Ibrido tramite LLM (3 pagine alla volta)
        final_text = llm_normalize_markdown(marker_text, r'\{\d+\}-+', pages_per_chunk=3)
        os.makedirs(base_output_dir, exist_ok=True)
        final_path = os.path.join(base_output_dir, base_name + ".md")
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        print(f"✅ Documento finale (Ibrido + Normalizzazione LLM) generato: {final_path}")
            
        return final_text
    return marker_text


def process_single_file(file_path, output_dir="output_dir", metadata=None): # Aggiunto metadata
    access_level = metadata.get("access_level") if metadata else None
    if not access_level:
        print(f"❌ Access level non trovato in {file_path}. Rifiuto il caricamento.")
        return []

    force_ocr = metadata.get("module_option") == "sì" # Estrai l'opzione per forzare l'OCR

    print(f"Elaboro file: {file_path}")
    # Passa l'opzione force_ocr a get_md_content
    content = get_md_content(file_path, output_dir, force_ocr=force_ocr)
    if not content:
        print(f"❌ Contenuto non disponibile per {file_path}. Salto.")
        return []


    total_pages = get_total_pages(file_path)
    file_url = os.path.abspath(file_path)
    file_name = os.path.basename(file_path)

    metadata.update({"source_path": file_path})

    file_chunks = run_semantic_chunking(
        content,
        metadata=metadata
    )

    if file_chunks:
        save_document_and_chunks_to_db(file_name, file_url, access_level, total_pages, file_chunks)
    return file_chunks


def get_md_content(file_path, output_dir="output_dir", force_ocr=False): # Aggiunto force_ocr
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        # Passa force_ocr alla funzione di elaborazione PDF
        return run_marker_pdf_with_fallback(file_path, output_dir, force_ocr=force_ocr)
    elif ext == ".docx":
        if force_ocr:
            print("⚠️ OCR forzata richiesta per DOCX. Attualmente supportata solo per PDF. Procedo con conversione standard.")
        return convert_docx_to_md(file_path, output_dir)
    elif ext == ".md":
        if force_ocr:
            print("⚠️ OCR forzata richiesta per MD. Non applicabile. Procedo con lettura standard.")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        text = ocr_single_image(file_path)
        # Salvataggio nella stessa cartella dell'immagine
        output_path = os.path.splitext(file_path)[0] + ".md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ OCR salvato in: {output_path}")
        return text
    else:
        print(f"Ignoro file non supportato: {file_path}")
        return None


def process_folder(folder_path, output_dir="output_dir", metadata=None):
    print(f"--- Elaborazione cartella: {folder_path} ---")
    folder_output_dir = os.path.join(output_dir, os.path.basename(folder_path))
    os.makedirs(folder_output_dir, exist_ok=True)
    all_chunks = []

    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext in {".pdf", ".docx", ".md", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
                file_chunks = process_single_file(file_path, folder_output_dir, metadata)
                all_chunks.extend(file_chunks)

    return all_chunks


def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_semantic_chunks(chunks, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    input_path = r"ocr_debug_images\page_7.jpg"  # Sostituisci con il percorso reale del tuo file o cartella
    output_dir = "output_dir"
    metadata = {"access_level": "public", "module_option": "no"}  # Sostituisci con la logica reale per estrarre l'access level
    ensure_output_dir(output_dir)

    if os.path.isdir(input_path):
        chunks = process_folder(input_path, output_dir, metadata)
    elif os.path.isfile(input_path):
        chunks = process_single_file(input_path, output_dir, metadata)
    else:
        raise FileNotFoundError(f"Il percorso specificato non esiste: {input_path}")

    if chunks:
        save_semantic_chunks(chunks, os.path.join(output_dir, "semantic_chunks.json"))
        print(f"✅ Embedding completato. Salvati {len(chunks)} chunk in {os.path.join(output_dir, 'semantic_chunks.json')}")
    else:
        print("⚠️ Nessun chunk generato.")
