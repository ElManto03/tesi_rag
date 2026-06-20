import os
import re
import numpy as np
from pypdf import PdfReader, PdfWriter
from pdf2image import convert_from_path
from PIL import Image
import io
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTChar

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

def check_ocr_basic_quality(text):
    """Controlli euristici non-AI sulla qualità dell'OCR."""
    if not text or len(text.strip()) < 15:
        return False
    # Controllo righe vuote eccessive o pattern di tabelle allucinate (es. | | | |)
    if re.search(r'(?:\|\s*){6,}\|', text):
        print("🔍 Qualità OCR: Fallito controllo loop o tabelle vuote.")
        return False
    return True

def is_marker_reliable(vision_text, marker_text):
    """Controlla se marker_text è affidabile confrontando i caratteri alfanumerici con vision_text."""
    if not marker_text: return False
    v_count = len(re.findall(r'\w', vision_text))
    m_count = len(re.findall(r'\w', marker_text))
    return m_count >= (v_count * 0.5) # Affidabile se ha almeno il 50% dei caratteri

def trova_punto_taglio_ottimale(img):
    """Individua la riga di pixel per il taglio orizzontale senza spezzare il testo."""
    img_gray = img.convert('L')
    width, height = img.size
    meta_altezza = height // 2
    arr = np.array(img_gray)
    threshold_consecutive = int(0.4 * width)
    is_black = arr < 100
    res_consecutive = is_black.copy()
    k_step = 1
    while k_step * 2 <= threshold_consecutive:
        res_consecutive[:, :-k_step] &= res_consecutive[:, k_step:]
        k_step *= 2
    remainder = threshold_consecutive - k_step
    if remainder > 0:
        res_consecutive[:, :-remainder] &= res_consecutive[:, remainder:]
    tutti_i_candidati = np.where(np.any(res_consecutive[:, :max(1, width - threshold_consecutive + 1)], axis=1))[0]

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

    if punto_taglio is not None:
        # Verifica limiti di sicurezza (se il taglio è nel primo o ultimo 15%, non tagliare)
        if punto_taglio < 0.15 * height or punto_taglio > 0.85 * height:
            return None
        return int(punto_taglio + 1)

    return None

def croppa_su_testo_reale(img, tolleranza_bianco=250, margine_sicurezza=30):
    """Rimuove i bordi vuoti o le linee della tabella inutili a destra."""
    img_data = np.array(img.convert("L"))
    h, w = img_data.shape
    pixel_scuri = img_data < tolleranza_bianco
    bordo_v = int(h * 0.05)
    taglio_x = w
    for x in range(w - 1, -1, -1):
        colonna_centrale = pixel_scuri[bordo_v : h - bordo_v, x]
        somma_pixel = np.sum(colonna_centrale)
        if 7 < somma_pixel < (h * 0.3):
            taglio_x = x + margine_sicurezza
            break
    return img.crop((0, 0, min(max(taglio_x, int(w * 0.6)), w), h))

    # -------------------------------------------------------------------------
# PIPELINE DI SICUREZZA: Rimozione Metadati e Controllo Prompt Injection
# -------------------------------------------------------------------------
def sterilizza_struttura_pdf(pdf_path_grezzo, pdf_path_pulito):
    """Rimuove metadati globali e annotazioni/commenti dalle pagine."""
    reader = PdfReader(pdf_path_grezzo)
    writer = PdfWriter()
    commenti_rimossi = 0

    for page in reader.pages:
        if "/Annots" in page:
            commenti_rimossi += len(page["/Annots"])
            del page["/Annots"]
        writer.add_page(page)
        
    writer.add_metadata({})
    with open(pdf_path_pulito, "wb") as f:
        writer.write(f)
        
    print(f"   [✓] Struttura: Rimossi {commenti_rimossi} commenti e azzerati i metadati.")

def analizza_geometria_e_sicurezza(pdf_path_pulito):
    """
    Scansiona il PDF carattere per carattere.
    - Rileva e scarta testo bianco o microscopico (invisibile all'utente).
    - Rimuove caratteri corrotti, non stampabili o con codifica non valida.
    - Esegue un controllo euristico anti-Prompt Injection sul testo estratto.
    """
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    
    testo_visibile_accumulato = []
    testo_invisibile_rilevato = False
    pagine_sospette = set()
    
    pattern_injection = [
        r"ignora le istruzioni", 
        r"ignore previous instructions", 
        r"da ora in poi agisci come", 
        r"tu sei un modello di ia", 
        r"system prompt bypass"
    ]

    with open(pdf_path_pulito, 'rb') as fp:
        for page_num, page in enumerate(PDFPage.get_pages(fp), start=1):
            interpreter.process_page(page)
            layout = device.get_result()
            
            pagina_corrente_sospetta = False
            
            for element in layout:
                if isinstance(element, LTTextBox):
                    for text_line in element:
                        if isinstance(text_line, LTTextLine):
                            testo_linea_sicuro = ""
                            
                            for car in text_line:
                                if isinstance(car, LTChar):
                                    char_text = car.get_text()
                                    # Ignoriamo spazi, tab e a capo dal conteggio dei sospetti
                                    is_real_char = bool(char_text.strip())
                                    
                                    # A) CONTROLLO DIMENSIONI (Testo Microscopico)
                                    if car.size < 2.0:
                                        if is_real_char:
                                            testo_invisibile_rilevato = True
                                            pagina_corrente_sospetta = True
                                        continue
                                    
                                    # B) CONTROLLO COLORE (Con tolleranza per float)
                                    if hasattr(car, 'graphicstate') and car.graphicstate.ncolor:
                                        colore = car.graphicstate.ncolor
                                        
                                        # Spazio Colore RGB: usiamo > 0.99 invece di == 1.0 per tolleranza
                                        if isinstance(colore, (list, tuple)) and len(colore) == 3:
                                            if float(colore[0]) > 0.99 and float(colore[1]) > 0.99 and float(colore[2]) > 0.99:
                                                if is_real_char:
                                                    testo_invisibile_rilevato = True
                                                    pagina_corrente_sospetta = True
                                                continue
                                        
                                        # Spazio Colore Scala di Grigi
                                        elif isinstance(colore, (int, float)) or (isinstance(colore, (list, tuple)) and len(colore) == 1):
                                            valore_grigio = colore[0] if isinstance(colore, (list, tuple)) else colore
                                            if float(valore_grigio) > 0.99:
                                                if is_real_char:
                                                    testo_invisibile_rilevato = True
                                                    pagina_corrente_sospetta = True
                                                continue
                                    
                                    # C) AGGIUNTO: CONTROLLO E FILTRO CARATTERI NON VALIDI / CORROTTI
                                    char_text = car.get_text()
                                    
                                    # Scartiamo caratteri vuoti o di controllo Unicode non stampabili (es. \x00, \u200b)
                                    # Conserviamo solo se è esplicitamente stampabile o se è uno spazio standard
                                    if not char_text.isprintable() and char_text not in " \t":
                                        # Non blocchiamo il programma, semplicemente ignoriamo questo carattere corrotto e andiamo avanti
                                        continue
                                        
                                    testo_linea_sicuro += char_text
                            
                            testo_pulito = " ".join(testo_linea_sicuro.split())
                            if testo_pulito:
                                testo_visibile_accumulato.append(testo_pulito)
            
            if pagina_corrente_sospetta:
                pagine_sospette.add(page_num - 1)
                                
    testo_finale_puro = "\n".join(testo_visibile_accumulato)
    
    # D) CONTROLLO ANOMALIE E PROMPT INJECTION
    if testo_invisibile_rilevato:
        print("   ⚠️ [ALERT] PDFMiner: Trovate e rimosse tracce di testo invisibile.")
        
    for pattern in pattern_injection:
        if re.search(pattern, testo_finale_puro.lower()):
            print(f"   🚨 [CRITICAL ALERT] Rilevato pattern di Prompt Injection: '{pattern}'")
            raise ValueError(f"Sicurezza Violata: Il documento contiene stringhe dannose di Prompt Injection.")
            
    print("   [✓] PDFMiner: Controllo geometrico, bonifica caratteri e analisi Injection superati.")
    return pagine_sospette

def sanitizza_txt_e_markdown(testo_grezzo):
    """
    Sanitizza il contenuto di file TXT e MD rimuovendo commenti HTML,
    caratteri invisibili e normalizzando gli spazi.
    """
    # 1. Rimuove i commenti HTML (tipici del Markdown, es: <!-- commento -->)
    # La regex r'<!--.*?-->' intercetta i tag di commento anche su più righe
    testo_senza_commenti = re.sub(r'<!--.*?-->', '', testo_grezzo, flags=re.DOTALL)
    
    # 2. Rimuove caratteri Unicode invisibili di controllo (es. Zero-Width Space \u200b)
    # Manteniamo solo i caratteri stampabili, i ritorni a capo (\n, \r) e le tabulazioni (\t)
    testo_pulito = "".join(ch for ch in testo_senza_commenti if ch.isprintable() or ch in "\n\r\t")
    
    # 3. Normalizzazione degli spazi bianchi
    # Rimuove spazi superflui all'inizio/fine di ogni riga e collassa gli spazi multipli
    linee = [ " ".join(linea.split()) for linea in testo_pulito.splitlines() ]
    
    # Restituiamo il testo ricostruito
    return "\n".join(linee).strip()

def rasterizza_pagine_pdf(input_pdf, output_pdf, indices):
    """Rasterizza le pagine specificate del PDF per rimuovere fisicamente il testo nascosto."""
    poppler_path = r"C:\Users\federico.mantoni\AppData\Local\miniconda3\envs\tesi2\Library\bin"
    images = convert_from_path(input_pdf, dpi=200, poppler_path=poppler_path)
    
    reader = PdfReader(input_pdf)
    writer = PdfWriter()
    
    for i in range(len(reader.pages)):
        if i in indices:
            img = images[i]
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PDF', resolution=200)
            img_byte_arr.seek(0)
            
            img_reader = PdfReader(img_byte_arr)
            writer.add_page(img_reader.pages[0])
        else:
            writer.add_page(reader.pages[i])
            
    with open(output_pdf, "wb") as f:
        writer.write(f)

def esegui_pipeline_sicurezza(percorso_pdf_grezzo):
    """Pipeline principale di sanitizzazione prima dell'elaborazione."""
    print(f"\n=== 🛡️ AVVIO PIPELINE DI SICUREZZA PER: {os.path.basename(percorso_pdf_grezzo)} ===")
    cartella = os.path.dirname(percorso_pdf_grezzo)
    nome_base = os.path.basename(percorso_pdf_grezzo)
    percorso_pulito = os.path.join(cartella, f"sanitized_{nome_base}")
    
    try:
        # 1. Rimozione metadati e annotazioni
        sterilizza_struttura_pdf(percorso_pdf_grezzo, percorso_pulito)
        
        # 2. Rilevamento pagine con testo nascosto
        pagine_sospette = analizza_geometria_e_sicurezza(percorso_pulito)
        
        if pagine_sospette:
            print(f"   ⚠️ [ALERT] Testo nascosto trovato nelle pagine: {[p+1 for p in pagine_sospette]}")
            print(f"   🛡️ Rasterizzazione in corso per neutralizzare le minacce...")
            percorso_rasterizzato = os.path.join(cartella, f"rasterized_{nome_base}")
            rasterizza_pagine_pdf(percorso_pulito, percorso_rasterizzato, pagine_sospette)
            os.replace(percorso_rasterizzato, percorso_pulito)

        print("=== 🎉 DOCUMENTO SANITIZZATO E SICURO ===\n")
        return percorso_pulito, pagine_sospette
    except ValueError as e:
        if os.path.exists(percorso_pulito):
            os.remove(percorso_pulito)
        raise e