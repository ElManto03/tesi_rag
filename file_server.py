import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

# Configura il percorso assoluto alla cartella dove sono memorizzati i documenti a scuola
DOCUMENTS_DIR = Path("C:/Users/federico.mantoni/Documents/tesi_rag/files").resolve()

router = APIRouter()

def genera_url_sicuro(file_name: str, base_url: str = "http://localhost:8000") -> str:
    """
    Prende il nome del file (o il percorso relativo) memorizzato nel DB
    e restituisce l'URL sicuro da passare al frontend.
    """
    if not file_name:
        return ""
    
    # Estraiamo solo il nome del file per evitare di esporre sottogestione di cartelle interne
    pure_name = os.path.basename(file_name)
    
    # Ritorna l'URL che punta al nostro endpoint sicuro
    return f"{base_url}/documenti/visualizza/{pure_name}"


@router.get("/documenti/visualizza/{file_name}")
async def visualizza_documento(file_name: str):
    """
    Endpoint protetto che valida il file e lo serve al browser.
    Se è un PDF, il browser lo aprirà nativamente invece di scaricarlo.
    """
    # ─── DISPOSITIVO DI SICUREZZA CONTRO PATH TRAVERSAL ───
    # Forziamo il percorso a risolversi all'interno della cartella autorizzata
    target_path = Path(DOCUMENTS_DIR / file_name).resolve()
    
    # Controllo di sicurezza fondamentale: il file finale si trova DAVVERO dentro DOCUMENTS_DIR?
    if not target_path.is_relative_to(DOCUMENTS_DIR):
        print(f"[-] Tentativo di violazione Path Traversal rilevato per il file: {file_name}")
        raise HTTPException(status_code=403, detail="Accesso non consentito al percorso specificato.")
    
    # Controllo esistenza
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Documento non trovato.")
    
    # Ritorna il file. 'inline' indica al browser di provare a visualizzarlo (se PDF) anziché scaricarlo
    return FileResponse(
        path=target_path,
        media_type="application/pdf",
        filename=target_path.name,
        content_disposition_type="inline"
    )