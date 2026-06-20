import os
import base64
from abc import ABC, abstractmethod
import requests
import subprocess
import shutil
import pypandoc
import json
import re
import io
import time
import numpy as np
import pandas as pd
import tempfile
import docx
from PIL import ImageEnhance, Image
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader, PdfWriter
from pdf2image import convert_from_path
from docling.document_converter import DocumentConverter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTChar

# Import LlamaIndex components for optimization
from llama_index.core import Settings, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.readers.docling import DoclingReader
from llama_index.multi_modal_llms.ollama import OllamaMultiModal
from llama_index.core.schema import ImageDocument

# Import moduli locali
from document_preprocessing import (
    clean_junk_text, check_ocr_basic_quality, is_marker_reliable,
    trova_punto_taglio_ottimale, croppa_su_testo_reale, esegui_pipeline_sicurezza,
    sanitizza_txt_e_markdown
)

# Configurazione cartella debug per MD
DEBUG_DIR = "ocr_debug_files"
os.makedirs(DEBUG_DIR, exist_ok=True)

# Configurazione per Marker e Ollama
PATH_TO_MARK = r"c:\Users\federico.mantoni\AppData\Local\miniconda3\envs\tesi2\Scripts\marker_single.exe"
os.environ.setdefault('PYPANDOC_PANDOC', 'C:/Users/federico.mantoni/AppData/Local/miniconda3/envs/tesi2/Library/bin/pandoc.exe')
os.environ["LLM_SERVICE"] = "openai" 
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
os.environ["OPENAI_API_KEY"] = "ollama" # Ollama non richiede chiave, ma Marker sì
os.environ["GEN_MODEL"] = "qwen2.5:14b-instruct-q4_K_M" # Modello generativo per Marker
CORRECTION_MODEL = "qwen2.5:14b-instruct-q4_K_M"

TABLE_OCR_PROMPT = """Tu sei un OCR visivo avanzato. Il tuo compito è convertire l'immagine in formato Markdown pulito, accurato e strutturato.

Segui queste istruzioni operative tassative:

1. TRASCRIZIONE INTEGRALE E LETTERALE
- VIETATO RIASSUMERE, PARAFRASARE O ACCORCIARE: Copia ogni singola parola, articolo e congiunzione carattere per carattere. Qualsiasi omissione è un errore grave.
- Mantieni la formattazione visiva (grassetto, sottolineato, MAIUSCOLO).
- ATTENZIONE AI PORTATI A CAPO A FINE RIGA: Se una frase o un punto elenco è lungo e si sviluppa su più righe visive sovrapposte all'interno dello stesso blocco, non spezzare le parole e non invertire l'ordine dei frammenti di testo. Leggi la riga superiore fino alla fine, poi scendi alla riga inferiore partendo da sinistra e unisci il testo in modo logico e continuo.

2. TRASCRIZIONE DI TUTTO IL TESTO
- Trascrivi tutto il testo sia fuori che dentro le tabelle

3. GESTIONE DEI CAMBI DI STRUTTURA E CELLE UNITE (Risoluzione dei blocchi)
- L'immagine può presentare tabelle miste: sezioni divise in più colonne che poi confluiscono in righe a colonna singola.
- Se una riga o una sezione della tabella occupa visivamente l'intera larghezza della pagina (cella unita / colspan), NON forzarla dentro la griglia a più colonne precedente. 
- Gestisci le righe a colonna singola o i blocchi uniti in questo modo:
  * Se contengono una sola parola o una dicitura di intestazione, chiudi la tabella precedente, digitala come un TITOLO in grassetto o testo normale centrato, e poi prosegui.
  * Se contengono un elenco di punti o caselle di controllo sotto un'intestazione, trascrivili come testo normale fuori dalla tabella, usando l'elenco puntato standard Markdown o i quadratini.
- GESTIONE DELLE CELLE MULTI-RIGA (ROW-SPAN): Se una colonna contiene un unico valore che si estende visivamente in verticale accanto a più righe della tabella, considera l'intero blocco come un'UNICA grande cella logica. Non spezzare il testo della colonna di destra in più righe Markdown separate; mantieni tutto associato allo stesso valore di sinistra nella stessa riga di tabella, usando il tag <br> per andare a capo all'interno delle celle se necessario.

4. REGOLE PER LE ZONE A TABELLA STANDARD (A più colonne)
- Nelle sezioni a più colonne, rispetta rigorosamente l'allineamento verticale tramite i caratteri |.
- Se una cella di una tabella a più colonne contiene più righe, frasi o quadratini, inserisci tutto nella STESSA cella separando i vettori ESCLUSIVAMENTE con il tag HTML <br>.
- NON generare righe interamente vuote nel codice Markdown.

5. REGOLE DI OUTPUT
- Restituisci ESCLUSIVAMENTE l'output in Markdown. No introduzioni, no commenti, no spiegazioni."""

PAGE_OCR_PROMPT = """Trascrivine il contenuto in Markdown standard.
Se trovi a inizio pagina una tabella con più colonne ma una sola di queste è non vuota, considerala come il continuo di una precedente tabella.
Se vedi parti di tabella che hanno una sola riga e colonna, trasformali in un elenco puntato.
Se vedi parti di tabella con una sola colonna e una sola parola al suo interno, trasformalo in un titolo.
Assicurati di scrivere tutto il testo nelle tabelle, se non riesci a dividere correttamente una riga di una tabella, trasformala in un paragrafo normale ma mantieni il testo.
Mantieni titoli e strutture. Solo testo Markdown, no commenti."""

TABLE_OCR_OPTIONS = {"num_ctx": 8192, "temperature": 0.0, "num_predict": 4096, "repeat_penalty": 1.3}
PAGE_OCR_OPTIONS = {"num_ctx": 16384, "num_predict": 4096, "temperature": 0.2, "repeat_penalty": 1.3}


#page 4 con ripetizioni passava
def check_ocr_ai_validation(text):
    """Controllo tramite LLM per frasi senza senso o allucinazioni (usato quando non c'è Marker source)."""
    validation_prompt = f"""Analizza il seguente testo estratto tramite OCR. 
Il testo contiene allucinazioni gravi, frasi senza senso, sequenze di caratteri casuali o incomplete o ripetizioni consecutive di uno stesso testo? 
Rispondi esclusivamente con la parola 'sì' se il testo contiene questi errori (ed è quindi da scartare), oppure 'no' altrimenti.

Testo da analizzare:
---
{text[:1500]}
---
Risposta (sì/no):"""

    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": CORRECTION_MODEL, "prompt": validation_prompt, "stream": False,
            "options": {"temperature": 0.0, "num_predict": 5}
        }, timeout=60)
        val = response.json().get("response", "").strip().lower()
        return not ("sì" in val or "si" in val)
    except: return True



#7 vuole skip a step 3
def refine_ocr_with_llm(vision_text, marker_text):
    """Fonde la struttura di VISION con la precisione testuale di MARKER."""
    print(f"--- 🛠️ Raffinamento strutturale con LLM ({CORRECTION_MODEL}) ---")
    prompt = f"""Tu sei un assistente specializzato nella ricostruzione e correzione di documenti scolastici. Il tuo compito è sistemare questa pagina che contiene una o più tabelle Markdown. Devi correggere eventuali errori fondendo le fonti che ti verranno fornite.

Ecco le tue fonti:
1. SORGENTE_VISION (Generata da un OCR visivo): Mantieni la STRUTTURA della tabella di questo testo se pensi sia ragionevole. Questo testo però potrebbe contenere frasi allucinate o incomplete da correggere. 
2. SORGENTE_TEXT (Generata da Marker-PDF): Ha una STRUTTURA probabilmente errata, ma il testo e le parole interne sono probabilmente integrali, corretti in italiano e privi di refusi di lettura.

Prima di fondere fai questi due controlli:
- Se tutto il testo in SORGENTE_TEXT è presente in SORGENTE_VISION e non trovi alcun testo diverso, e la struttura della tabella di SORGENTE_VISION non è assurda, non fare modifiche e scrivi "corretto" come risposta.
- Se c'è del testo di SORGENTE_VISION incompleto, e SORGENTE_TEXT non contiene testo con cui poter correggerlo, allora interrompi tutto e scrivi come risposta solo "skip".

Se il testo fuori dalle tabelle in SORGENTE_VISION non ha problemi, mantienilo così com'è.

Istruzioni per la fusione delle tabelle:
- Usa la SORGENTE_VISION come mappa per capire quante colonne e quante righe creare.
- Usa la SORGENTE_TEXT per "riempire" le celle, sostituendo le parole allucinate o i refusi della sorgente vision con il testo pulito e integro di Marker-PDF.
- Se nella SORGENTE_VISION un intero blocco di testo è stato ruotato o trasposto (es. trasformato in intestazione), raddrizzalo usando il senso logico delle frasi complete che trovi nella SORGENTE_TEXT.

Restituisci esclusivamente la pagina Markdown senza commenti o introduzioni.

---
SORGENTE_VISION:
{vision_text}

---
SORGENTE_TEXT:
{marker_text}
"""
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": CORRECTION_MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 12288}
        }, timeout=300)
        res_text = response.json().get("response", "").strip()
        #llm = Ollama(model=CORRECTION_MODEL, base_url=os.environ["OLLAMA_API_BASE"], request_timeout=300, additional_kwargs={"num_ctx": 12288})
        #res_text = str(llm.complete(prompt)).strip()
        res_clean = res_text.lower().rstrip('.')
        if res_clean == "corretto":
            print("✨ L'LLM ha confermato che l'OCR originale è corretto. Salto il raffinamento.")
            return vision_text
        if res_clean == "skip":
            print("⚠️ L'LLM ha richiesto di saltare questo tentativo (skip).")
            return None
        return res_text
    except: return vision_text

def _call_normalization_llm(chunk_text):
    """Normalizza blocchi di markdown tramite LLM."""
    #print(f"DEBUG: Chiamata normalizzazione su chunk di {len(chunk_text)} caratteri")
    correction_prompt = f"""
Tu sei un sistema di post-processing algoritmico per motori di Document Layout Analysis (DLA). 
Ricevi in input un frammento di testo (circa 3 pagine) in formato Markdown estratto da un documento. A causa dei cambi pagina, le tabelle potrebbero apparire spezzate o frammentate.

Il tuo obiettivo è normalizzare, ricostruire e consolidare la struttura logica del documento, restituendo un Markdown pulito e privo di rumore, ottimizzato per un sistema RAG.

Segui tassativamente queste linee guida ingegneristiche e universali:

1. IDENTIFICAZIONE E FUSIONE DELLE TABELLE SPEZZATE:
- Se una tabella si interrompe a causa di un cambio pagina e riprende poco dopo (riconoscibile dalla ripetizione delle stesse intestazioni di colonna o dalla continuazione logica delle righe), fondile in un'unica tabella Markdown.
- Elimina le intestazioni duplicate e i separatori di tabella (`|---|---|`) intermedi generati dal cambio pagina, ma lascia i separatori di pagina di tipo {{n}}------

2. RISOLUZIONE DEI TRONCAMENTI SINTATTICI (SALDATURA DEL TESTO):
- Analizza la fine di ogni riga e l'inizio della successiva. Se una parola o una frase è chiaramente troncata a metà da un a capo, salda i due frammenti eliminando i trattini di sillabazione.

3. PROPAGAZIONE DELLE CELLE UNITE VERTICALMENTE (SPANNING):
- Se rilevi che una cella contiene dati strutturati importanti e le righe immediatamente successive presentano celle corrispondenti vuote (o contenenti solo spazi/simboli come '//'), propaga logicamente quel contesto. Duplica il testo nelle righe successive.

4. ELIMINAZIONE DEL RUMORE DI PAGINA (STRIPPING):
- Riconosci e rimuovi completamente metadati ripetitivi che interrompono il flusso logico: numeri di pagina, intestazioni di pagina (header), piè di pagina (footer), indirizzi istituzionali, stringhe di protocollo e tag di annotazione delle immagini (es. ``).

5. PRESERVAZIONE DEI SEPARATORI DI PAGINA:
- **IMPORTANTE**: I separatori di pagina nel formato `{{N}}---` (es. `{{1}}---`, `{{2}}---`) sono marcatori strutturali essenziali. **Devono essere preservati** nella loro posizione originale. Non rimuoverli e non modificarli. Se unisci una tabella tra 2 pagine, puoi spostare

6. PRESERVAZIONE DEL TESTO NON TABELLARE:
- Se il documento contiene paragrafi di testo normale, preservali esattamente nella loro posizione sequenziale originaria.

7. RIMOZIONE DELLE INTESTAZIONI:
- All'inizio di ogni blocco di testo, se trovi un'intestazione che contiene il numero di telefono o il codice fiscale della scuola (82007870403), rimuovila completamente insieme a eventuali righe adiacenti che contengono informazioni di contatto o indirizzi.

OUTPUT RICHIESTO:
Restituisci esclusivamente l'intero documento normalizzato in Markdown. Non includere alcuna introduzione, alcuna spiegazione e nessun commento discorsivo (es. NON scrivere "Ecco la tabella strutturata:"). Inizia direttamente con il primo elemento valido del documento.

Ecco il testo grezzo da elaborare:
{chunk_text}
"""
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": CORRECTION_MODEL, "prompt": correction_prompt, "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 12288}
        }, timeout=300)
        return response.json().get("response", "").strip()
        #llm = Ollama(model=CORRECTION_MODEL, base_url=os.environ["OLLAMA_API_BASE"], request_timeout=300, additional_kwargs={"num_ctx": 12288})
        #return str(llm.complete(correction_prompt)).strip()
    except: return chunk_text

def llm_normalize_markdown(md_text, separator_pattern, pages_per_chunk=3):
    """Divide il markdown e normalizza a blocchi."""
    print(f"--- 🚀 Normalizzazione finale Markdown con LLM ({CORRECTION_MODEL}) ---")
    md_text = re.sub(r'<!--\s*image\s*-->', '', md_text)
    matches = list(re.finditer(separator_pattern, md_text, re.MULTILINE))
    if not matches: return _call_normalization_llm(md_text)
    final_text = ""
    if matches[0].start() > 0:
        print("🔧 Normalizzazione testo iniziale...")
        final_text += _call_normalization_llm(md_text[:matches[0].start()]) + "\n\n"
    for i in range(0, len(matches), pages_per_chunk):
        start_pos = matches[i].start()
        print(f"🔧 Normalizzazione blocco di pagine a partire da {matches[i].group(0).strip()}...")
        end_pos = matches[i + pages_per_chunk].start() if i + pages_per_chunk < len(matches) else len(md_text)
        corrected = _call_normalization_llm(md_text[start_pos:end_pos])
        #final_text += matches[i].group(0) + "\n\n" + corrected + "\n\n"
        chunk_to_correct = md_text[start_pos:end_pos]
        
        # Estrai i separatori originali presenti nel chunk
        original_separators = re.findall(separator_pattern, chunk_to_correct, re.MULTILINE)
        
        corrected = _call_normalization_llm(chunk_to_correct)
        
        # Fallback: se l'LLM ha rimosso i separatori, li reinseriamo noi
        for sep in original_separators:
            if sep not in corrected:
                # Inseriamo il separatore all'inizio se manca il primo, o cerchiamo di dedurre la posizione
                # Per semplicità e robustezza, lo mettiamo all'inizio del blocco corretto.

                corrected = sep + "\n\n" + corrected

        final_text += corrected + "\n\n"
    return final_text

class OCRProvider(ABC):
    @abstractmethod
    def generate_text(self, img_str: str) -> str:
        pass

class OllamaOCRAdapter(OCRProvider):
    def __init__(self, base_url=None):
        self.base_url = base_url or "http://localhost:11434"
        self.models = ["qwen2.5vl:3b", "qwen2.5vl:7b"]

    def _encode(self, pil_img):
        target_width = 1000
        if pil_img.width > target_width:
            h_size = int((float(pil_img.height) * float(target_width / float(pil_img.width))))
            pil_img = pil_img.resize((target_width, h_size), Image.Resampling.LANCZOS)
        padded = Image.new(pil_img.mode, (pil_img.width, pil_img.height + 60), "white")
        padded.paste(pil_img, (0, 30))
        enhancer = ImageEnhance.Contrast(padded.convert('L'))
        optimized = enhancer.enhance(1.4)
        buf = io.BytesIO()
        optimized.save(buf, format='JPEG', quality=95)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def _call(self, img_b64, prompt, options, model):
        mm_llm = OllamaMultiModal(model=model, base_url=self.base_url, request_timeout=300, additional_kwargs={"options": options})
        # Wrap the base64 image in an ImageDocument for LlamaIndex MultiModal logic
        res = requests.post(f"{self.base_url}/api/generate", json={
            "model": model, "prompt": prompt, "images": [img_b64], "stream": False, "options": options
        }, timeout=300)
        return clean_junk_text(res.json().get("response", ""))
        #img_doc = ImageDocument(image=img_b64, image_mimetype="image/jpeg")
        #res = mm_llm.complete(prompt=prompt, image_documents=[img_doc])
        #return clean_junk_text(str(res))

    def generate_text(self, pil_img, prompt=PAGE_OCR_PROMPT, options=None, marker_text=None, page_idx=None):
        if options is None: options = PAGE_OCR_OPTIONS
        img_b64 = self._encode(pil_img)
        best_fallback = None

        def save_debug_md(content, suffix):
            if page_idx is not None:
                path = os.path.join(DEBUG_DIR, f"page_{page_idx+1}_{suffix}.md")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
        
        # Tentativo 1: 3b intero
        print("🚀 Tentativo 1: OCR immagine intera (modello 3b)...")
        try:
            res = self._call(img_b64, prompt, options, "qwen2.5vl:3b")
            if check_ocr_basic_quality(res):
                if best_fallback is None: best_fallback = res
                if marker_text and is_marker_reliable(res, marker_text):
                    refined = refine_ocr_with_llm(res, marker_text)
                    if refined:
                        save_debug_md(res, "raw")
                        save_debug_md(refined, "refined")
                        return refined
                elif check_ocr_ai_validation(res):
                    save_debug_md(res, "raw")
                    print("✅ OCR 3b completato con successo.")
                    return res
        except Exception as e:
            print(f"⚠️ Errore Tentativo 1: {e}")

        # Tentativo 2: Taglio + 3b/7b
        try:
            riga = trova_punto_taglio_ottimale(pil_img)
            if riga:
                print(f"✂️ Immagine troppo grande o qualità bassa. Eseguo taglio intelligente alla riga {riga}...")
                parts = [pil_img.crop((0,0,pil_img.width,riga)), pil_img.crop((0,riga,pil_img.width,pil_img.height))]
                for model in self.models:
                    print(f"🚀 Tentativo OCR su parti spezzate (modello {model})...")
                    try:
                        combined = ""
                        for p in parts:
                            combined += self._call(self._encode(croppa_su_testo_reale(p)), prompt, options, model) + "\n\n"
                        if check_ocr_basic_quality(combined):
                            if best_fallback is None: best_fallback = combined
                            if marker_text and is_marker_reliable(res, marker_text):
                                refined = refine_ocr_with_llm(combined, marker_text)
                                if refined:
                                    save_debug_md(combined, "raw")
                                    save_debug_md(refined, "refined")
                                    return refined
                            elif check_ocr_ai_validation(combined):
                                save_debug_md(combined, "raw")
                                return combined
                    except Exception as e:
                        print(f"⚠️ Errore Tentativo 2 ({model}): {e}")
                        continue
        except Exception as e:
            print(f"⚠️ Errore durante il calcolo del taglio: {e}")
        
        # Fallback finale: 7b intero
        print("🚀 Tentativo finale: OCR immagine intera (modello 7b)...")
        try:
            res = self._call(img_b64, prompt, options, "qwen2.5vl:7b")
            if check_ocr_basic_quality(res):
                if best_fallback is None: best_fallback = res
                if marker_text and is_marker_reliable(res, marker_text):
                    refined = refine_ocr_with_llm(res, marker_text)
                    if refined:
                        save_debug_md(res, "raw")
                        save_debug_md(refined, "refined")
                        return refined
                elif check_ocr_ai_validation(res):
                    save_debug_md(res, "raw")
                    return res
        except Exception as e:
            print(f"❌ Errore critico nel tentativo finale: {e}")

        if best_fallback:
            print("📦 Tutti i tentativi avanzati falliti o andati in timeout. Restituisco il primo risultato che ha superato la qualità base.")
            save_debug_md(best_fallback, "raw")
            return best_fallback
            
        return ""

def get_pages_with_tables(base_output_dir):
    """Analizza blocks.json di Marker per trovare pagine con tabelle, usando euristiche avanzate."""
    blocks_path = os.path.join(base_output_dir, "blocks.json")
    if not os.path.exists(blocks_path):
        return []
    
    table_pages = set()
    list_group_pages = set()
    page_top_level_blocks = {} # page_id -> primi blocchi della pagina

    def find_data_recursive(blocks):
        for block in blocks:
            b_type = str(block.get("block_type", "")).lower()
            b_desc = str(block.get("block_description", "")).lower()
            pid = block.get("page_id")
            
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
            
            # Esplora ricorsivamente i figli se presenti
            children = block.get("children")
            if isinstance(children, list):
                find_data_recursive(children)

    try:
        with open(blocks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
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
        
    return sorted(list(table_pages))
    

class VLLMOCRAdapter(OCRProvider):
    def __init__(self, model_name, api_url):
        self.model_name = model_name
        self.api_url = api_url

    def generate_text(self, img_str):
        # Placeholder per chiamata OpenAI-compatible a vLLM Vision
        print(f"🚀 [vLLM] OCR tramite endpoint {self.api_url}")
        return "[Testo OCR da vLLM]"


def get_total_pages(file_path):
    """Restituisce il numero totale di pagine di un file PDF."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        try:
            reader = PdfReader(file_path)
            return len(reader.pages)
        except Exception:
            return None
    return None


def convert_to_md_with_docling(file_path, output_dir):
    """Converte un file (DOCX, HTML, ecc.) in Markdown usando Docling."""
    print(f"--- 🚀 Avvio Docling: {file_path} ---")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, base_name + ".md")

    try:
        reader = DoclingReader()
        docs = reader.load_data(file_path=file_path)
        md_text = "\n\n".join([d.get_content() for d in docs])
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_text)
            
        print(f"✅ Conversione Docling completata: {output_path}")
        return md_text
    except Exception as e:
        print(f"❌ Errore durante la conversione con Docling: {e}")
        return None

def run_marker_pdf(pdf_path, output_dir):
    """Esegue Marker-PDF."""
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    base_output_dir = os.path.join(output_dir, base_name)
    nested_md = os.path.join(base_output_dir, base_name + ".md")
    try:
        subprocess.run([PATH_TO_MARK, pdf_path, "--output_dir", output_dir, "--paginate_output", "--debug"], check=True)
        if os.path.exists(nested_md):
            with open(nested_md, "r", encoding="utf-8") as f: return f.read()
    except: pass
    return None

def glm_ocr(pdf_path, target_page_indices=None, ocr_provider=None, marker_text=None):
    """Esegue Vision-OCR avanzato con logica di raffinamento."""
    print(f"--- 🚨 Avvio pipeline Vision-OCR (Qwen2.5-VL) ---")
    if ocr_provider is None: ocr_provider = OllamaOCRAdapter()
    poppler_path = r"C:\Users\federico.mantoni\AppData\Local\miniconda3\envs\tesi2\Library\bin"
    
    marker_pages = {}
    if marker_text:
        parts = re.split(r'\{(\d+)\}-+', marker_text)
        for i in range(1, len(parts), 2): marker_pages[int(parts[i])] = parts[i+1].strip()

    if target_page_indices:
        pages = convert_from_path(pdf_path, 200, first_page=min(target_page_indices)+1, last_page=max(target_page_indices)+1, poppler_path=poppler_path)
        target_pages = [(idx, pages[i]) for i, idx in enumerate(range(min(target_page_indices), max(target_page_indices)+1)) if idx in target_page_indices]
    else:
        pages = convert_from_path(pdf_path, 200, poppler_path=poppler_path)
        target_pages = list(enumerate(pages))

    prompt = TABLE_OCR_PROMPT if target_page_indices else PAGE_OCR_PROMPT
    options = TABLE_OCR_OPTIONS if target_page_indices else PAGE_OCR_OPTIONS
    
    results = {}
    for i, page in target_pages:
        print(f"📄 Elaborazione pagina {i+1}...")
        # Crop header/footer
        w, h = page.size
        cropped = page.crop((0, int(h * 0.10), w, int(h * 0.95)))

        results[i] = ocr_provider.generate_text(cropped, prompt=prompt, options=options, marker_text=marker_pages.get(i), page_idx=i)
        print(f"✅ Pagina {i+1} completata.")
        
    if target_page_indices: return results
    return "\n\n".join([f"{{{k}}}---" + v for k,v in sorted(results.items())])


def assemble_hybrid_markdown(marker_md, ocr_pages_dict):
    """Sostituisce le pagine testuali di Marker con l'output Vision OCR dove necessario."""
    page_pattern = r'(\{\d+\}-+)'
    parts = re.split(page_pattern, marker_md)
    
    assembled = [parts[0]]
    processed_indices = set()
    
    current_page_idx = -1
    
    for i in range(1, len(parts), 2):
        tag = parts[i]
        content = parts[i+1] if (i+1) < len(parts) else ""
        
        page_match = re.search(r'\{(\d+)\}', tag)
        if page_match:
            current_page_idx = int(page_match.group(1))
            processed_indices.add(current_page_idx)
            
        if current_page_idx in ocr_pages_dict:
            print(f"🔄 Ibridazione: Sostituzione pagina {current_page_idx + 1} con output Vision OCR.")
            assembled.append(tag + "\n" + ocr_pages_dict[current_page_idx])
        else:
            assembled.append(tag + content)

    missing_pages = sorted([idx for idx in ocr_pages_dict if idx not in processed_indices])
    if missing_pages:
        print(f"⚠️ Attenzione: {len(missing_pages)} pagine OCR non avevano un tag corrispondente nel testo Marker.")

    for idx in missing_pages:
        print(f"➕ Ibridazione: Aggiunta pagina mancante {idx + 1} con output Vision OCR.")
        assembled.append(f"\n\n{{{idx}}}------------------------------------------------\n" + ocr_pages_dict[idx])
            
    return "".join(assembled)


def run_marker_pdf_with_fallback(pdf_path, output_dir, force_ocr=False):
    """
    Esegue Marker su PDF, con fallback a Vision-OCR se il risultato è insufficiente
    o se l'OCR è forzata. Include logica ibrida per tabelle.
    """
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    base_output_dir = os.path.join(os.path.abspath(output_dir), base_name)
    
    # --- SANITIZZAZIONE E SICUREZZA ---
    try:
        pdf_path, pagine_sospette = esegui_pipeline_sicurezza(pdf_path)
    except ValueError:
        print(f"❌ File {pdf_path} scartato per violazione sicurezza.")
        return ""

    if force_ocr:
        res = llm_normalize_markdown(glm_ocr(pdf_path), r'\{\d+\}-+')
        os.makedirs(base_output_dir, exist_ok=True)
        with open(os.path.join(base_output_dir, base_name + ".md"), "w", encoding="utf-8") as f:
            f.write(res)
        return res

    marker_text = run_marker_pdf(pdf_path, output_dir)
    
    if not marker_text or len(marker_text) < 100:
        print("⚠️ Marker ha prodotto un risultato insufficiente. Attivo fallback Vision...")
        if os.path.exists(base_output_dir):
            try:
                shutil.rmtree(base_output_dir) # Pulisci output Marker problematico
            except Exception as e:
                print(f"⚠️ Errore eliminazione directory Marker: {e}")
        
        result_text = glm_ocr(pdf_path)
        os.makedirs(base_output_dir, exist_ok=True)
        with open(os.path.join(base_output_dir, base_name + ".md"), "w", encoding="utf-8") as f:
            f.write(res)
        return res
    else:
        # tables = get_pages_with_tables(base_output_dir)
        # if tables:
        #     print(f"📊 Rilevate tabelle nelle pagine: {tables}. Avvio OCR pagina per pagina...")
        #     ocr_results = glm_ocr(pdf_path, target_page_indices=tables, marker_text=marker_text)
        #     hybrid = llm_normalize_markdown(assemble_hybrid_markdown(marker_text, ocr_results), r'\{\d+\}-+')
            # Logica Ibrida: cerca tabelle e aggiunge le pagine sospette per forzare l'OCR
        table_pages = set(get_pages_with_tables(base_output_dir))
        if pagine_sospette:
            print(f"🛡️ Rilevate {len(pagine_sospette)} pagine sospette (testo nascosto). Forzo l'OCR per queste pagine.")
            table_pages.update(pagine_sospette)

        if table_pages:
            sorted_pages = sorted(list(table_pages))
            print(f"📊 Avvio OCR mirato per le pagine: {sorted_pages}.")
            ocr_results = glm_ocr(pdf_path, target_page_indices=sorted_pages, marker_text=marker_text)
            hybrid = llm_normalize_markdown(assemble_hybrid_markdown(marker_text, ocr_results), r'\{\d+\}-+')
            with open(os.path.join(base_output_dir, base_name + ".md"), "w", encoding="utf-8") as f: f.write(hybrid)
            return hybrid
    return marker_text 

# ---- DEBUG
def ocr_single_image(image_path):
    """Esegue l'OCR su un singolo file immagine e restituisce il testo Markdown."""
    print(f"🖼️ Esecuzione OCR su immagine: {image_path}")
    adapter = OllamaOCRAdapter()
    if isinstance(image_path, str):
        with Image.open(image_path) as img:
            return adapter.generate_text(img, prompt=TABLE_OCR_PROMPT, options=TABLE_OCR_OPTIONS)
    else:
        return adapter.generate_text(image_path, prompt=TABLE_OCR_PROMPT, options=TABLE_OCR_OPTIONS)
# ---- DEBUG

def sanitizza_e_leggi_csv(csv_path):
    # Leggiamo il CSV con Pandas
    df = pd.read_csv(csv_path)
    
    # Sanitizzazione: se una cella inizia con un carattere di formula (=, +, -, @), 
    # lo disinneschiamo aggiungendo un apice singolo davanti, trasformandolo in testo puro.
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).apply(
            lambda x: f"'{x}" if x.startswith(('=', '+', '-', '@')) else x
        )
    
    # Convertiamo il dataframe in un formato testuale pulito (es. Markdown) 
    # così Docling o il tuo Chunker possono elaborarlo come testo strutturato
    return df.to_markdown(index=False)

def get_md_content(file_path, output_dir="output_dir", force_ocr=False, debug_page=None):
    """
    Determina il tipo di file e restituisce il suo contenuto in formato Markdown,
    applicando conversioni o OCR se necessario.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        # ---- DEBUG
        if debug_page is not None:
            print(f"🔍 DEBUG: Esecuzione OCR mirata sulla pagina {debug_page}...")
            page_idx = int(debug_page) - 1
            ocr_results = glm_ocr(file_path, target_page_indices=[page_idx])
            res = ocr_results.get(page_idx, "")
            debug_out = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0])
            os.makedirs(debug_out, exist_ok=True)
            with open(os.path.join(debug_out, f"pag.{debug_page}.md"), "w", encoding="utf-8") as f:
                f.write(res)
            return res
        # ---- DEBUG
        return run_marker_pdf_with_fallback(file_path, output_dir, force_ocr=force_ocr)
    elif ext == ".docx":
        if force_ocr:
            print("⚠️ OCR forzata richiesta per DOCX. Attualmente supportata solo per PDF. Procedo con conversione standard.")
        return convert_to_md_with_docling(file_path, output_dir)

    elif ext == ".md" or ext == ".txt":
        if force_ocr:
            print("⚠️ OCR forzata richiesta per MD. Non applicabile. Procedo con lettura standard.")
        with open(file_path, "r", encoding="utf-8") as f:
            return sanitizza_txt_e_markdown(f.read())
    elif ext == ".csv":
        return sanitizza_e_leggi_csv(file_path)

    # ---- DEBUG
    elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        text = ocr_single_image(file_path)
        output_path = os.path.splitext(file_path)[0] + ".md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ OCR salvato in: {output_path}")
        return text
    # ---- DEBUG
    else:
        print(f"Ignoro file non supportato: {file_path}")
        return None