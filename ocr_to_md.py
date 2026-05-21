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
import psycopg
import io
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader
from pdf2image import convert_from_path, pdfinfo_from_path
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from llama_index.core.node_parser import SemanticSplitterNodeParser, MarkdownNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Document, Settings
from config import settings

def custom_sentence_splitter(text):
    return LegalSegmenter().split_sentences(text)

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
os.environ["GEN_MODEL"] = "llama3.1:8b" #
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

    # --- PULIZIA TESTO (Rimozione boilerplate scuola) ---
    # Definiamo i pattern per identificare le righe di intestazione ripetute
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
        is_junk = False
        if line.strip():
            is_junk = any(re.match(p, line.strip(), re.IGNORECASE) for p in junk_patterns)
        
        if not is_junk:
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

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
        # Cerca il tag id="page-N" nel contenuto del chunk
        match = re.search(r'id="page-(\d+)', content)
        if match:
            current_page = int(match.group(1))
        
        # Iniettiamo il metadato direttamente nel nodo LlamaIndex
        node.metadata["page_number"] = current_page

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
            "--paginate_output" # Disabilitiamo l'OCR interno di Marker per affidarlo completamente a GLM-OCR in caso di fallback
            # "--use_llm",
            # "--llm_service", "marker.services.ollama.OllamaService",
            # "--ollama_base_url", os.environ["OLLAMA_API_BASE"],
            # "--ollama_model", "qwen2.5vl:7b"
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

def glm_ocr(pdf_path):
    print(f"--- 🚨 Avvio Fallback GLM-OCR ---")
    poppler_path = r"C:\Users\federico.mantoni\AppData\Local\miniconda3\envs\tesi2\Library\bin"
    # Ridotto a 200 DPI: ottimo compromesso tra velocità e precisione per Qwen
    pages = convert_from_path(pdf_path, 200, poppler_path=poppler_path, thread_count=4)
    
    def process_page(i, page):
        # Convertiamo in base64 in memoria senza passare dal disco
        img_byte_arr = io.BytesIO()
        page.save(img_byte_arr, format='JPEG', quality=85)
        img_str = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5vl:7b",
                    "prompt": (
                        "Agisci come un esperto OCR. Trascrivi il testo di questa pagina. "
                        "REGOLE: 1. Tabelle in Markdown standard (| colonna |). "
                        "2. Mantieni i titoli (#, ##). 3. Solo testo, no commenti."
                    ),
                    "images": [img_str],
                    "stream": False,
                    "options": {
                        "num_gpu": 99,
                        "num_ctx": 4096,
                        "temperature": 0
                    }
                },
                timeout=120
            )
            res_json = response.json()
            testo_pagina = res_json.get("response", "")
            if not testo_pagina and "message" in res_json:
                testo_pagina = res_json["message"].get("content", "")
            
            print(f"✅ Pagina {i+1} completata")
            return i, testo_pagina
        except Exception as e:
            print(f"❌ Errore pagina {i+1}: {e}")
            return i, f"[Errore OCR Pagina {i+1}]"

    # Parallelizziamo le richieste (max_workers=2 per le tue 2 GPU)
    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_page, i, page) for i, page in enumerate(pages)]
        for future in futures:
            results.append(future.result())

    # Ordiniamo i risultati per indice per mantenere l'ordine delle pagine
    results.sort(key=lambda x: x[0])
    full_text = "\n\n".join([r[1] for r in results])
    return full_text.strip()

def run_marker_pdf_with_fallback(pdf_path, output_dir, force_ocr=False):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    base_output_dir = os.path.join(os.path.abspath(output_dir), base_name)
    
    if force_ocr:
        print("--- 🚨 OCR forzata: Avvio GLM-OCR ---")
        # Se l'OCR è forzata, bypassa Marker e vai direttamente a glm_ocr
        result_text = glm_ocr(pdf_path)
        # Salva il risultato del fallback anche quando forzato
        fallback_dir = os.path.join(output_dir, base_name)
        os.makedirs(fallback_dir, exist_ok=True)
        fallback_path = os.path.join(fallback_dir, base_name + ".md")
        with open(fallback_path, "w", encoding="utf-8") as f:
            f.write(result_text)
        return result_text

    # Prova prima con Marker
    result_text = run_marker_pdf(pdf_path, output_dir)
    
    # Se Marker fallisce o restituisce meno di 100 caratteri (testo troppo corto per un modulo)
    if not result_text or len(result_text) < 100:
        print("⚠️ Marker ha prodotto un risultato insufficiente. Attivo fallback Vision...")
        
        # Elimina l'intera directory di output di Marker se il fallback viene attivato
        if os.path.exists(base_output_dir):
            try:
                shutil.rmtree(base_output_dir)
                print(f"🗑️ Eliminato output di Marker: {base_output_dir}")
            except Exception as e:
                print(f"⚠️ Errore eliminazione directory Marker: {e}")
        
        result_text = glm_ocr(pdf_path)
        
        # Salva il risultato del fallback
        fallback_dir = os.path.join(output_dir, base_name)
        os.makedirs(fallback_dir, exist_ok=True)
        fallback_path = os.path.join(fallback_dir, base_name + ".md")
        with open(fallback_path, "w", encoding="utf-8") as f:
            f.write(result_text)
            
    return result_text


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
            if ext in {".pdf", ".docx", ".md"}:
                file_chunks = process_single_file(file_path, folder_output_dir, metadata)
                all_chunks.extend(file_chunks)

    return all_chunks


def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_semantic_chunks(chunks, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)

execute_all = False

if __name__ == "__main__":
    input_path = r"files\Regolamento-disciplinare.pdf"  # Sostituisci con il percorso reale del tuo file o cartella
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
