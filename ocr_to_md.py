import os
import subprocess
#from pdf2image import convert_from_path, pdfinfo_from_path
#from glmocr import GlmOcr

def run_marker_pdf_light(pdf_path, output_dir):
    """
    Versione ultra-leggera di Marker per macchine senza GPU.
    """
    print(f"--- Tentativo con Marker-PDF (Light Mode): {pdf_path} ---")
    try:
        # Usiamo parametri che riducono il carico di memoria
        subprocess.run([
            "marker_single", 
            pdf_path, 
            "--output_dir", output_dir,
            "--debug",
            "--use_llm"
        ], check=True)
        
        base_name = os.path.basename(pdf_path).replace(".pdf", "")
        expected_md = os.path.join(output_dir, base_name, f"{base_name}.md")
        
        if os.path.exists(expected_md):
            with open(expected_md, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        print(f"Marker-PDF troppo pesante o non riuscito: {e}")
    return None

# def run_glm_ocr_cpu_friendly(pdf_path, config_path):
#     """
#     Fallback OCR: Processa una pagina alla volta per salvare la RAM.
#     """
#     print(f"--- Fallback su GLM-OCR (CPU Mode) ---")
    
#     # Otteniamo il numero di pagine senza caricarle tutte
#     info = pdfinfo_from_path(pdf_path)
#     total_pages = info["Pages"]
    
#     md_results = []

#     with GlmOcr(config_path=config_path) as parser:
#         for i in range(1, total_pages + 1):
#             # Carichiamo SOLO la pagina corrente
#             page = convert_from_path(pdf_path, dpi=150, first_page=i, last_page=i)[0]
#             temp_image = f"temp_page_{i}.png"
#             page.save(temp_image, "PNG")
            
#             print(f"Analisi OCR pagina {i}/{total_pages} (Lenta su CPU)...")
            
#             try:
#                 # Nota: assicurati che in config.yaml il timeout sia alto (>300s)
#                 result = parser.parse(temp_image)
                
#                 text = ""
#                 if hasattr(result, 'markdown_result'):
#                     text = result.markdown_result
#                 elif hasattr(result, 'markdown'):
#                     text = result.markdown
#                 else:
#                     text = str(result)
                
#                 md_results.append(f"## Pagina {i}\n\n{text}")
#             except Exception as e:
#                 print(f"Errore alla pagina {i}: {e}")
#             finally:
#                 if os.path.exists(temp_image):
#                     os.remove(temp_image)
                    
#     return "\n\n---\n\n".join(md_results)

def smart_convert_16gb(pdf_path, output_md_path, config_ocr="config.yaml"):
    output_dir = "marker_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Su 16GB, Marker potrebbe fallire subito se non hai SWAP attivo.
    content = run_marker_pdf_light(pdf_path, output_dir)
    
    if not content or content.strip() == "":
        print("Marker fallito. Passo all'OCR a basso consumo...")
        #content = run_glm_ocr_cpu_friendly(pdf_path, config_ocr)

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"--- Completato! File: {output_md_path} ---")

if __name__ == "__main__":
    smart_convert_16gb("174__VERONA_LAGO_DI_GARDA_-MANTOVA__viaggio_distuzione-_CLASSI_2A_2B_2C_BATTISTI.pdf", "crash-pc.md")