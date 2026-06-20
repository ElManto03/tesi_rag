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
import hashlib
import numpy as np
import time
from datetime import datetime
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
os.makedirs(DEBUG_DIR, exist_ok=True)
for f in os.listdir(DEBUG_DIR):
    if f.startswith("optimized"):
        os.remove(os.path.join(DEBUG_DIR, f))

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

def trova_punto_taglio_ottimale(img_or_path):
    """
    Trova la riga di pixel dell'immagine che contiene più del 40% di pixel neri 
    più vicina alla metà della pagina. Assume l'immagine in B/N (0=nero).
    """
    if isinstance(img_or_path, str):
        img = Image.open(img_or_path)
    else:
        img = img_or_path

    img = img.convert('L')
    width, height = img.size
    meta_altezza = height // 2
    arr = np.array(img)
    
    # 1. Identificazione righe con pixel neri consecutivi > 40% della riga
    threshold_consecutive = int(0.4 * width)
    is_black = arr < 100
    
    # Algoritmo di raddoppio per trovare sequenze di lunghezza N in log(N) passi
    res_consecutive = is_black.copy()
    k_step = 1
    while k_step * 2 <= threshold_consecutive:
        res_consecutive[:, :-k_step] &= res_consecutive[:, k_step:]
        k_step *= 2
    remainder = threshold_consecutive - k_step
    if remainder > 0:
        res_consecutive[:, :-remainder] &= res_consecutive[:, remainder:]
    
    has_long_sequence = np.any(res_consecutive[:, :max(1, width - threshold_consecutive + 1)], axis=1)
    tutti_i_candidati = np.where(has_long_sequence)[0]

    # Fallback: calcoliamo la densità totale per la ricerca della riga più nera se non ci sono linee lunghe
    percentuale_nero = np.mean(is_black, axis=1)
    
    punto_taglio = None
    
    # 2. Filtriamo i candidati per quelli sotto la metà
    indici_sotto_meta = tutti_i_candidati[tutti_i_candidati > meta_altezza]
    if len(indici_sotto_meta) > 0:
        candidato = np.min(indici_sotto_meta)
        # Se il punto selezionato sotto fa parte dell'ultimo 10%, cerchiamo invece sopra
        if candidato < 0.9 * height:
            punto_taglio = candidato

    if punto_taglio is None:
        # 3. Cerchiamo la riga più vicina alla metà partendo da sopra tra i candidati
        indici_sopra_meta = tutti_i_candidati[tutti_i_candidati <= meta_altezza]
        if len(indici_sopra_meta) > 0:
            punto_taglio = np.max(indici_sopra_meta)

    # 4. Ottimizzazione del punto di taglio risalendo blocchi contigui e cercando spazi bianchi
    set_candidati = set(tutti_i_candidati)

    # def risali_blocco(idx):
    #     """Risale le righe contigue che superano la soglia del 40% di nero."""
    #     while (idx - 1) in set_candidati:
    #         idx -= 1
    #     return idx

    # if punto_taglio is not None and punto_taglio in set_candidati:
    #     # Risaliamo il blocco del candidato attuale (procedura di sicurezza)
    #     punto_taglio_r = risali_blocco(punto_taglio)

    #     # Verifichiamo se esiste un candidato precedente separato da spazio bianco
    #     pos_idx = np.searchsorted(tutti_i_candidati, punto_taglio_r)
    #     if pos_idx > 0:
    #         candidato_prec = tutti_i_candidati[pos_idx - 1]
    #         candidato_prec = risali_blocco(candidato_prec)
    #         if candidato_prec >= 10:
    #             blocco_sopra = percentuale_nero[candidato_prec - 25 : candidato_prec]
    #             if np.all(blocco_sopra < 0.05):
    #                 # Se troviamo spazio bianco, prendiamo questo candidato e risaliamo il suo blocco
    #                 punto_taglio = candidato_prec

    # # 5. Verifica limiti di sicurezza (se il taglio è nel primo o ultimo 15%, non tagliare)
    # if punto_taglio is not None:
    #     if punto_taglio < 0.15 * height or punto_taglio > 0.85 * height:
    #         return None
            
    # return int(punto_taglio+1) if punto_taglio is not None else None

def croppa_su_testo_reale(img_or_path, tolleranza_bianco=250, margine_sicurezza=30):
    """
    Taglia l'immagine dove finisce il testo reale, ignorando le linee esterne
    della tabella.
    """
    if isinstance(img_or_path, str):
        img = Image.open(img_or_path)
    else:
        img = img_or_path
        
    img_gray = img.convert("L")
    img_data = np.array(img_gray)

    h, w = img_data.shape

    # 1. Trova i pixel scuri
    pixel_scuri = img_data < tolleranza_bianco

    # 2. TRUCCO: Per ignorare le linee perimetrali superiore e inferiore,
    # non analizziamo il 5% in alto e il 5% in basso dell'immagine.
    bordo_v = int(h * 0.05)
    
    taglio_x = w
    
    # Scansioniamo da destra a sinistra
    for x in range(w - 1, -1, -1):
        # Prendiamo la colonna escludendo i bordi sopra e sotto
        colonna_centrale = pixel_scuri[bordo_v : h - bordo_v, x]
        
        # Calcoliamo la densità: una linea verticale isolata (il bordo destro)
        # colora quasi tutta la colonna. Il testo invece colora solo pochi pixel nella colonna.
        somma_pixel = np.sum(colonna_centrale)
        
        if 7 < somma_pixel < (h * 0.3):
            taglio_x = x + margine_sicurezza
            break

    # Evitiamo di rompere i limiti dell'immagine
    taglio_x = min(max(taglio_x, int(w * 0.6)), w)

    return img.crop((0, 0, taglio_x, h))

def pipeline_taglio_intelligente(image_path, altezza_soglia_critica=1440):
    """
    Decide se tagliare l'immagine e, in caso positivo, esegue il taglio
    in un punto sicuro senza tranciare il testo.
    """
    if isinstance(image_path, str):
        img = Image.open(image_path)
    else:
        img = image_path

    width, height = img.size
    
    # CRITERIO DEL SE: Tagliamo solo se supera la soglia critica per la VRAM
    if height <= altezza_soglia_critica:
        print(f"Immagine sicura (Altezza: {height}px). Nessun taglio necessario.")
        return [img], False # Restituisce l'immagine intera e False per "was_split"
    
    print(f"Immagine troppo grande ({height}px). Ricerca del punto di taglio sicuro...")
    
    # CRITERIO DEL DORE: Trova la riga di testo vuota
    riga_taglio = trova_punto_taglio_ottimale(img)
    if riga_taglio is None:
        print("Taglio annullato: riga di taglio non trovata o troppo vicina ai bordi (limite 15%). Procedo con l'immagine intera.")
        return [img], False # Restituisce l'immagine intera e False per "was_split"
    print(f"Punto di taglio sicuro individuato alla riga: {riga_taglio}")
    
    # Eseguiamo il crop anatomico
    meta_top = img.crop((0, 0, width, riga_taglio))
    meta_bottom = img.crop((0, riga_taglio, width, height))

    # Applichiamo il crop orizzontale intelligente per ignorare i bordi tabella
    meta_top = croppa_su_testo_reale(meta_top)
    meta_bottom = croppa_su_testo_reale(meta_bottom)

    meta_top.save("meta_top.png", format='PNG', quality=95, optimize=True)
    meta_bottom.save("meta_bottom.png", format='PNG', quality=95, optimize=True)
    return [meta_top, meta_bottom], True # Restituisce le parti tagliate e True per "was_split"

def encode_image(img_or_path):
    """Ottimizza e codifica un oggetto PIL Image in base64."""
    if isinstance(img_or_path, str):
        img = Image.open(img_or_path)
    else:
        img = img_or_path

    # target_width = 1176
    # if img.width > target_width:
    #     w_percent = (target_width / float(img.width))
    #     h_size = int((float(img.height) * float(w_percent)))
    #     w_safe = (target_width + 3) & ~3
    #     h_safe = (h_size + 3) & ~3
    #     img = img.resize((w_safe, h_safe), Image.Resampling.LANCZOS)


    target_width = 1000
    if img.width > target_width:
        w_percent = (target_width / float(img.width))
        h_size = int((float(img.height) * float(w_percent)))
        img = img.resize((target_width, h_size), Image.Resampling.LANCZOS)

    #Aggiunge 30 righe di pixel bianchi sopra l'immagine
    w, h = img.size
    padded_img = Image.new(img.mode, (w, h + 60), "white")
    padded_img.paste(img, (0, 30))
    img = padded_img

    grayscale_img = img.convert('L')
    enhancer = ImageEnhance.Contrast(grayscale_img)
    optimized_img = enhancer.enhance(1.4)
    img_byte_arr = io.BytesIO()
    optimized_img.save(img_byte_arr, format='JPEG', quality=95, optimize=True)
    #return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    # Genera un percorso univoco per il salvataggio
    save_path = os.path.join(DEBUG_DIR, f"optimized_{int(time.time() * 1000)}.png")

    # Salva l'immagine ottimizzata
    optimized_img.save(save_path, format='PNG', quality=95, optimize=True)
    return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    


# --- CONFIGURAZIONE OCR OLLAMA ---
TABLE_OCR_OPTIONS = {"num_ctx": 8192, "temperature": 0.0, "num_predict": 4096, "repeat_penalty": 1.3}
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
- Se rilevi che una cella contiene dati strutturati importanti e le righe immediatamente successive presentano celle corrispondenti vuote (o contenenti solo spazi/simboli come '//'), propaga logicamente quel contesto. Duplica il testo nelle righe successive.

4. ELIMINAZIONE DEL RUMORE DI PAGINA (STRIPPING):
- Riconosci e rimuovi completamente metadati ripetitivi che interrompono il flusso logico: numeri di pagina, intestazioni di pagina (header), piè di pagina (footer), indirizzi istituzionali, stringhe di protocollo e tag di annotazione delle immagini (es. ``).

5. PRESERVAZIONE DEL TESTO NON TABELLARE:
- Se il documento contiene paragrafi di testo normale, preservali esattamente nella loro posizione sequenziale originaria.

6. RIMOZIONE DELLE INTESTAZIONI:
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
                "options": {"num_ctx": 12288, "temperature": 0.1}
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

#Provare a cambiare testo prima colonna dicendo di allungare riga sotto
#- PRIMA COLONNA: Assicurati di trascrivere TUTTO il testo nella colonna di sinistra, ma mantieni la struttura visiva originale.
#- GESTIONE DELLE CELLE UNITE VERTICALMENTE: Se una macro-categoria si applica a un intero blocco di elementi, crea un'unica grande riga Markdown. Inserisci tutti gli elementi corrispondenti nella cella adiacente separati da <br>, in modo che rimangano perfettamente allineati visivamente con le rispettive colonne di riferimento.
# 2. VIETATO TRASPORRE O RUOTARE LA TABELLA: Rispetta rigorosamente l'orientamento visivo originale. Le colonne dell'immagine DEVONO rimanere colonne nel Markdown (allineate in verticale tramite i caratteri |). Le righe dell'immagine DEVONO rimanere righe in orizzontale. È tassativamente vietato convertire i testi della colonna sinistra in intestazioni di colonna.
TABLE_OCR_PROMPT = """Tu sei un OCR visivo avanzato. Il tuo compito è convertire l'immagine in formato Markdown pulito, accurato e strutturato.

Segui queste istruzioni operative tassative:

1. TRASCRIZIONE INTEGRALE E LETTERALE
- VIETATO RIASSUMERE, PARAFRASARE O ACCORCIARE: Copia ogni singola parola, articolo e congiunzione carattere per carattere. Qualsiasi omissione è un errore grave.
- Mantieni la formattazione visiva (grassetto, sottolineato, MAIUSCOLO).
- ATTENZIONE AI PORTATI A CAPO A FINE RIGA: Se una frase o un punto elenco è lungo e si sviluppa su più righe visive sovrapposte all'interno dello stesso blocco, non spezzare le parole e non invertire l'ordine dei frammenti di testo. Leggi la riga superiore fino alla fine, poi scendi alla riga inferiore partendo da sinistra e unisci il testo in modo logico e continuo.

2. GESTIONE DEI CAMBI DI STRUTTURA E CELLE UNITE (Risoluzione dei blocchi)
- L'immagine può presentare tabelle miste: sezioni divise in più colonne che poi confluiscono in righe a colonna singola.
- Se una riga o una sezione della tabella occupa visivamente l'intera larghezza della pagina (cella unita / colspan), NON forzarla dentro la griglia a più colonne precedente. 
- Gestisci le righe a colonna singola o i blocchi uniti in questo modo:
  * Se contengono una sola parola o una dicitura di intestazione, chiudi la tabella precedente, digitala come un TITOLO in grassetto o testo normale centrato, e poi prosegui.
  * Se contengono un elenco di punti o caselle di controllo sotto un'intestazione, trascrivili come testo normale fuori dalla tabella, usando l'elenco puntato standard Markdown o i quadratini.
- GESTIONE DELLE CELLE MULTI-RIGA (ROW-SPAN): Se una colonna contiene un unico valore che si estende visivamente in verticale accanto a più righe della tabella, considera l'intero blocco come un'UNICA grande cella logica. Non spezzare il testo della colonna di destra in più righe Markdown separate; mantieni tutto associato allo stesso valore di sinistra nella stessa riga di tabella, usando il tag <br> per andare a capo all'interno delle celle se necessario.

3. REGOLE PER LE ZONE A TABELLA STANDARD (A più colonne)
- Nelle sezioni a più colonne, rispetta rigorosamente l'allineamento verticale tramite i caratteri |.
- Se una cella di una tabella a più colonne contiene più righe, frasi o quadratini, inserisci tutto nella STESSA cella separando i vettori ESCLUSIVAMENTE con il tag HTML <br>.
- NON generare righe interamente vuote nel codice Markdown.

4. REGOLE DI OUTPUT
- Restituisci ESCLUSIVAMENTE l'output in Markdown. No introduzioni, no commenti, no spiegazioni."""
PAGE_OCR_PROMPT = """Trascrivine il contenuto in Markdown standard.
Se trovi a inizio pagina una tabella con più colonne ma una sola di queste è non vuota, considerala come il continuo di una precedente tabella.
Se vedi parti di tabella che hanno una sola riga e colonna, trasformali in un elenco puntato.
Se vedi parti di tabella con una sola colonna e una sola parola al suo interno, trasformalo in un titolo.
Assicurati di scrivere tutto il testo nelle tabelle, se non riesci a dividere correttamente una riga di una tabella, trasformala in un paragrafo normale ma mantieni il testo.
Mantieni titoli e strutture. Solo testo Markdown, no commenti."""

def _call_ollama_ocr(image, prompt, options, model='qwen2.5vl:7b'):
    """Centralizza la chiamata a Ollama per ridurre la duplicazione e gestire la VRAM."""
    print(f"Inizio chiamata OCR Ollama ({model})...")
    headers = {"Connection": "close"}
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "images": image if isinstance(image, list) else [image],
            "options": options,
            "stream": False
        },
        headers=headers, timeout=300
    )
    response.raise_for_status()
    text = response.json().get("response", "").strip()
    if not text:
        raise Exception(f"Il modello {model} ha restituito un output vuoto.")

    requests.post("http://localhost:11434/api/generate", json={"model": model, "keep_alive": 0})
    return clean_junk_text(text)

def check_ocr_basic_quality(text):
    """Controlli euristici non-AI sulla qualità dell'OCR."""
    if not text or len(text.strip()) < 15:
        return False
    # Ripetizioni di righe vuote (più di 4 consecutive) spesso segno di loop
    if re.search(r'(\n\s*){5,}', text):
        print("🔍 Qualità OCR: Fallito controllo righe vuote ripetute.")
        return False
    return True

def check_ocr_ai_validation(text):
    """Controllo tramite LLM per frasi senza senso o allucinazioni (usato quando non c'è Marker source)."""
    validation_prompt = f"""Analizza il seguente testo estratto tramite OCR. 
Il testo contiene allucinazioni gravi, frasi totalmente senza senso, sequenze di caratteri casuali o ripetizioni ossessive? 
Rispondi esclusivamente con la parola 'sì' se il testo contiene questi errori (ed è quindi da scartare), oppure 'no' se il testo è leggibile e corretto.

Testo da analizzare:
---
{text[:1500]}
---
Risposta (sì/no):"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:14b-instruct-q4_K_M", # Modello leggero e veloce per la validazione
                "prompt": validation_prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 5}
            },
            timeout=60
        )
        val = response.json().get("response", "").strip().lower()
        if "sì" in val or "si" in val:
            print(f"🔍 Qualità OCR: LLM ha rilevato errori o no-sense (Risposta: {val}).")
            return False
        return True
    except Exception as e:
        print(f"⚠️ Errore validazione LLM: {e}. Procedo per prudenza.")
        return True

def refine_ocr_with_llm(vision_text, marker_text):
    """Raffina l'output Vision OCR usando il testo integrale di Marker-PDF come riferimento."""
    print(f"--- 🛠️ Raffinamento LLM ({CORRECTION_MODEL}) ---")
    prompt = f"""Tu sei un assistente specializzato nella ricostruzione e correzione di documenti scolastici. Il tuo compito è generare un'unica tabella Markdown finale perfetta partendo da due fonti che hanno difetti diversi.

Ecco le tue fonti:
1. SORGENTE_VISION (Generata da un OCR visivo): Ha un'OTTIMA struttura geometrica di tabelle e colonne, ma contiene allucinazioni nei testi, parole frullate (es. "FRECOLASTICA") e lettere mancanti.
2. SORGENTE_TEXT (Generata da Marker-PDF): Ha una STRUTTURA PESSIMA o disallineata, ma il testo e le parole interne sono al 100% integrali, corretti in italiano e privi di refusi di lettura.

Istruzioni per la fusione:
- Usa la SORGENTE_VISION come mappa per capire quante colonne e quante righe creare.
- Usa la SORGENTE_TEXT per "riempire" le celle, sostituendo le parole allucinate o i refusi della sorgente vision con il testo pulito e integro di Marker-PDF.
- Se nella SORGENTE_VISION un intero blocco di testo è stato ruotato o trasposto (es. trasformato in intestazione), raddrizzalo usando il senso logico delle frasi complete che trovi nella SORGENTE_TEXT.

Restituisci esclusivamente la tabella Markdown finale corretta, senza commenti o introduzioni.

---
SORGENTE_VISION:
{vision_text}

---
SORGENTE_TEXT:
{marker_text}
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": CORRECTION_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 12288}
            },
            timeout=300
        )
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"⚠️ Errore raffinamento LLM: {e}")
        return vision_text

def run_ocr_pipeline(img_or_path, prompt, options, marker_text_page=None, page_idx=None):
    """Pipeline di fallback: split + 3b -> split + 7b -> full + 7b con raffinamento LLM opzionale."""

    def save_debug_md(content, suffix):
        if page_idx is not None:
            path = os.path.join(DEBUG_DIR, f"page_{page_idx+1}_{suffix}.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

    if isinstance(img_or_path, str):
        img = Image.open(img_or_path)
    else:
        img = img_or_path

    # Tentativo 1: Immagine intera con modello 3b
    try:
        print("🚀 Tentativo 1: Immagine intera con modello 3b...")
        img_str = encode_image(img)
        res = _call_ollama_ocr(img_str, prompt, options, model="qwen2.5vl:3b")
        if check_ocr_basic_quality(res):
            save_debug_md(res, "raw")
            if marker_text_page:
                refined = refine_ocr_with_llm(res, marker_text_page)
                save_debug_md(refined, "refined")
                return refined
            elif check_ocr_ai_validation(res):
                return res
    except Exception as e:
        print(f"⚠️ Tentativo 1 (full+3b) fallito: {e}. Provo a valutare il taglio...")

    # Calcolo del taglio (soglia abbassata per favorire la frammentazione se l'intera fallisce)
    parts, was_split = pipeline_taglio_intelligente(img, altezza_soglia_critica=300)

    if was_split:
        # Tentativo 2: Immagine spezzata con modello 3b
        try:
            print("🚀 Tentativo 2: Immagine spezzata con modello 3b...")
            ocr_results = []
            for p in parts:
                img_str = encode_image(p)
                ocr_results.append(_call_ollama_ocr(img_str, prompt, options, model="qwen2.5vl:3b"))
            combined_res = "\n\n".join(ocr_results).strip()
            if check_ocr_basic_quality(combined_res):
            save_debug_md(combined_res, "raw")
                if marker_text_page:
                refined = refine_ocr_with_llm(combined_res, marker_text_page)
                save_debug_md(refined, "refined")
                return refined
                elif check_ocr_ai_validation(combined_res):
                    return combined_res
        except Exception as e:
            print(f"⚠️ Tentativo 2 (split+3b) fallito: {e}. Provo split+7b...")

        # Tentativo 3: Immagine spezzata con modello 7b
        try:
            print("🚀 Tentativo 3: Immagine spezzata con modello 7b...")
            ocr_results = []
            for p in parts:
                img_str = encode_image(p)
                ocr_results.append(_call_ollama_ocr(img_str, prompt, options, model="qwen2.5vl:7b"))
            combined_res = "\n\n".join(ocr_results).strip()
            if check_ocr_basic_quality(combined_res):
            save_debug_md(combined_res, "raw")
                if marker_text_page:
                refined = refine_ocr_with_llm(combined_res, marker_text_page)
                save_debug_md(refined, "refined")
                return refined
                elif check_ocr_ai_validation(combined_res):
                    return combined_res
        except Exception as e:
            print(f"⚠️ Tentativo 3 (split+7b) fallito: {e}. Provo fallback finale full+7b...")

    # Tentativo 4: Immagine intera con modello 7b (fallback finale)
    try:
        print("🚀 Tentativo 4 (Ultima spiaggia): Immagine intera con modello 7b...")
        img_str = encode_image(img)
        res = _call_ollama_ocr(img_str, prompt, options, model="qwen2.5vl:7b")
        if check_ocr_basic_quality(res):
            save_debug_md(res, "raw")
            if marker_text_page:
                refined = refine_ocr_with_llm(res, marker_text_page)
                save_debug_md(refined, "refined")
                return refined
            return res
    except Exception as e:
        print(f"❌ Tentativo finale (full+7b) fallito: {e}")
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

def save_document_and_chunks_to_db(file_name, file_url, access_level, total_pages, chunks, file_hash=None, md_hash=None, created_at=None, expiry_date=None):
    if not access_level:
        raise ValueError("Access level mancante; rifiuto il caricamento del documento.")

    try:
        with db_engine.begin() as conn:
            document_id = conn.execute(
                text(
                    "INSERT INTO documents (file_name, upload_date, file_url, access_level, total_pages, file_hash, md_hash, expiry_date) "
                    "VALUES (:file_name, :upload_date, :file_url, :access_level, :total_pages, :file_hash, :md_hash, :expiry_date) RETURNING id"
                ),
                {
                    "file_name": file_name,
                    "upload_date": created_at,
                    "file_url": file_url,
                    "access_level": access_level,
                    "total_pages": total_pages,
                    "file_hash": file_hash,
                    "md_hash": md_hash,
                    "expiry_date": expiry_date                
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
nelk                    }
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

def glm_ocr(pdf_path, target_page_indices=None, marker_text=None):
    print(f"--- 🚨 Avvio Vision-OCR (Qwen2.5-VL) ---")
    poppler_path = r"C:\Users\federico.mantoni\AppData\Local\miniconda3\envs\tesi2\Library\bin"

    marker_pages = {}
    if marker_text:
        page_pattern = r'\{(\d+)\}-+'
        parts = re.split(page_pattern, marker_text)
        for i in range(1, len(parts), 2):
            pid = int(parts[i])
            marker_pages[pid] = parts[i+1].strip()
    
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

        prompt = TABLE_OCR_PROMPT if target_indices_provided else PAGE_OCR_PROMPT

        testo_pagina = run_ocr_pipeline(page, prompt, current_options, marker_text_page=marker_pages.get(i), page_idx=i)
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
    return run_ocr_pipeline(image_path, TABLE_OCR_PROMPT, TABLE_OCR_OPTIONS)

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

    #Logica Ibrida: cerca tabelle e usa OCR mirato per quelle pagine
    table_pages, _ = get_table_data(base_output_dir)
    if table_pages:
        print(f"📊 Rilevate tabelle nelle pagine: {sorted(list(table_pages))}. Avvio OCR pagina per pagina...")
        ocr_results = glm_ocr(pdf_path, target_page_indices=table_pages, marker_text=marker_text)
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


def process_single_file(file_path, output_dir="output_dir", metadata=None, debug_page=None): # Aggiunto metadata
    access_level = metadata.get("access_level") if metadata else None
    if not access_level:
        print(f"❌ Access level non trovato in {file_path}. Rifiuto il caricamento.")
        return []

    force_ocr = metadata.get("module_option") == "sì" # Estrai l'opzione per forzare l'OCR
    expiry_date = metadata.get("expiry_date") # Estrai la data di scadenza opzionale

    print(f"Elaboro file: {file_path}")
    # Passa l'opzione force_ocr a get_md_content
    content = get_md_content(file_path, output_dir, force_ocr=force_ocr, debug_page=debug_page)
    if not content:
        print(f"❌ Contenuto non disponibile per {file_path}. Salto.")
        return []

    total_pages = get_total_pages(file_path)
    file_url = os.path.abspath(file_path)
    file_name = os.path.basename(file_path)
    
    created_at = datetime.now().isoformat()
    # --- CALCOLO METADATI E SALVATAGGIO FILE .META ---
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    md_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    # Determina la cartella di output (coerente con get_md_content)
    base_name = os.path.splitext(file_name)[0]
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        meta_save_dir = os.path.join(os.path.abspath(output_dir), base_name)
    else:
        meta_save_dir = os.path.abspath(output_dir)
    
    os.makedirs(meta_save_dir, exist_ok=True)
    meta_file_path = os.path.join(meta_save_dir, f"{file_name}.meta")
    
    meta_data = {
        "file_name": file_name,
        "file_url": file_url,
        "total_pages": total_pages,
        "file_hash": file_hash,
        "md_hash": md_hash,
        "ocr_used": force_ocr, # Nota: per precisione estrema andrebbe tracciato se l'ibrido ha attivato l'OCR
        "marker_version": "1.10.2",
        "created_at": created_at,
        "expiry_date": expiry_date
    }
    
    try:
        with open(meta_file_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=4, ensure_ascii=False)
        print(f"📝 Metadata salvati in: {meta_file_path}")
    except Exception as e:
        print(f"⚠️ Errore salvataggio metadata: {e}")
    # ------------------------------------------------

    metadata.update({"source_path": file_path})

    file_chunks = run_semantic_chunking(
        content,
        metadata=metadata
    )

    if file_chunks:
        save_document_and_chunks_to_db(
            file_name, file_url, access_level, total_pages, file_chunks,
            file_hash=file_hash, md_hash=md_hash,
            created_at=created_at, expiry_date=expiry_date
        )
    return file_chunks


def get_md_content(file_path, output_dir="output_dir", force_ocr=False, debug_page=None): # Aggiunto force_ocr
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        if debug_page is not None:
            # Se debug_page è presente nei metadati, eseguiamo l'OCR solo su quella pagina
            print(f"🔍 DEBUG: Esecuzione OCR mirata sulla pagina {debug_page}...")
            page_idx = int(debug_page) - 1
            ocr_results = glm_ocr(file_path, target_page_indices=[page_idx])
            with open(fr"output_dir/pag.{debug_page}.md", "w", encoding="utf-8") as f:
                f.write(ocr_results.get(page_idx, ""))
            return ocr_results.get(page_idx, "")

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
    input_path = r"files/Regolamento-disciplinare.pdf" 
    output_dir = "output_dir"
    metadata = {"access_level": "public", "module_option": "no"}  # Sostituisci con la logica reale per estrarre l'access level
    ensure_output_dir(output_dir)

    if os.path.isdir(input_path):
        chunks = process_folder(input_path, output_dir, metadata)
    elif os.path.isfile(input_path):
        #pag.7 avrebbe bisogno di split
        #pag.8 non ma con 7b
        chunks = process_single_file(input_path, output_dir, metadata, debug_page=14)  # Aggiunto debug_page per testare una pagina specifica
    else:
        raise FileNotFoundError(f"Il percorso specificato non esiste: {input_path}")

    if chunks:
        save_semantic_chunks(chunks, os.path.join(output_dir, "semantic_chunks.json"))
        print(f"✅ Embedding completato. Salvati {len(chunks)} chunk in {os.path.join(output_dir, 'semantic_chunks.json')}")
    else:
        print("⚠️ Nessun chunk generato.")
