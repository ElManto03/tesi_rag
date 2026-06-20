import os
import hashlib
import json
from datetime import datetime

# Import conversion logic from document_converter.py
from document_converter import (
    run_marker_pdf_with_fallback,
    convert_to_md_with_docling,
    get_total_pages,
    glm_ocr,
    ocr_single_image,
    get_md_content
)

# Import chunking logic from chunking_embedding.py
from chunking_embedding import run_semantic_chunking
from db_manager import save_document_and_chunks_to_db

def process_single_file(file_path, output_dir="output_dir", metadata=None, debug_page=None, db_role='user'):
    if(db_role == 'user'):
        raise Exception("Non hai il permesso per eseguire questa azione")
    access_level = metadata.get("access_level") if metadata else None
    if not access_level:
        print(f"❌ Access level not found in {file_path}. Refusing to upload.")
        return []

    force_ocr = metadata.get("module_option") == "sì" # Extract option to force OCR
    expiry_date = metadata.get("expiry_date") # Extract optional expiry date

    print(f"Processing file: {file_path}")
    content = get_md_content(file_path, output_dir, force_ocr=force_ocr, debug_page=debug_page)
    if not content:
        print(f"❌ Content not available for {file_path}. Skipping.")
        return []

    total_pages = get_total_pages(file_path)
    file_url = os.path.abspath(file_path)
    file_name = os.path.basename(file_path)
    
    created_at = datetime.now().isoformat()
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    md_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    
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
        "ocr_used": force_ocr,
        "marker_version": "1.10.2", # This should probably be dynamic or a constant
        "created_at": created_at,
        "expiry_date": expiry_date
    }
    
    try:
        with open(meta_file_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=4, ensure_ascii=False)
        print(f"📝 Metadata saved to: {meta_file_path}")
    except Exception as e:
        print(f"⚠️ Error saving metadata: {e}")

    metadata.update({"source_path": file_path})

    file_chunks = run_semantic_chunking(
        content,
        metadata=metadata
    )

    if file_chunks:
        save_document_and_chunks_to_db(
            file_name, file_url, access_level, total_pages, file_chunks,
            file_hash=file_hash, md_hash=md_hash,
            created_at=created_at, expiry_date=expiry_date,
            db_role=db_role
        )
    return file_chunks

def process_folder(folder_path, output_dir="output_dir", metadata=None, db_role='user'):
    print(f"--- Processing folder: {folder_path} ---")
    if(db_role == 'user'):
        raise Exception("Non hai il permesso per eseguire questa azione")

    folder_output_dir = os.path.join(output_dir, os.path.basename(folder_path))
    os.makedirs(folder_output_dir, exist_ok=True)
    all_chunks = []

    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext in {".pdf", ".docx", ".html", ".md", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
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
    input_path = r"files\REGOLAMENTO-DISTITUTO_AS202526.pdf" 
    output_dir = "output_dir"
    metadata = {"access_level": "public", "module_option": "no"}  # Sostituisci con la logica reale per estrarre l'access level
    ensure_output_dir(output_dir)

    if os.path.isdir(input_path):
        chunks = process_folder(input_path, output_dir, metadata)
    elif os.path.isfile(input_path):
        #pag.7 avrebbe bisogno di split
        #pag.8 non ma con 7b
        chunks = process_single_file(input_path, output_dir, metadata, db_role='developer')  # Aggiunto debug_page per testare una pagina specifica
    else:
        raise FileNotFoundError(f"Il percorso specificato non esiste: {input_path}")

    if chunks:
        save_semantic_chunks(chunks, os.path.join(output_dir, "semantic_chunks.json"))
        print(f"✅ Embedding completato. Salvati {len(chunks)} chunk in {os.path.join(output_dir, 'semantic_chunks.json')}")
    else:
        print("⚠️ Nessun chunk generato.")