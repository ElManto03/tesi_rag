from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.multi_modal_llms.ollama import OllamaMultiModal

# 1. Configura l'LLM (Qwen 30B) in Docker
Settings.llm = Ollama(
    model="llama3.2:3b", 
    base_url="http://localhost:11434",
    request_timeout=600.0 # Timeout lungo per modelli pesanti
)

# 2. Configura gli Embedding (mxbai-embed-large) in Docker
Settings.embed_model = OllamaEmbedding(
    model_name="bge-m3:latest",
    base_url="http://localhost:11434"
)

ocr = OllamaMultiModal(
    model="glm-ocr:latest", # Sostituisci con il nome esatto del modello che hai su Ollama
    base_url="http://localhost:11434",
    request_timeout=300.0
)

import os
from pathlib import Path
from typing import Dict, Any, List

def _get_embedding_from_model(model, text: str):
    """
    Chiamata robusta al metodo embedding esposto dal modello.
    Prova più nomi di metodo comuni (compatibilità con diverse versioni di LlamaIndex).
    """
    if hasattr(model, "get_text_embedding"):
        return model.get_text_embedding(text)
    if hasattr(model, "get_query_embedding"):
        return model.get_query_embedding(text)
    if hasattr(model, "get_embeddings"):
        # spesso accetta lista di stringhe
        out = model.get_embeddings([text])
        return out[0] if isinstance(out, (list, tuple)) else out
    if hasattr(model, "embed"):
        return model.embed(text)
    # fallback: se il model è una funzione-callable
    if callable(model):
        return model(text)
    raise RuntimeError("Il modello di embedding non espone un metodo riconosciuto (get_text_embedding/get_embeddings/embed).")

def embed_text(text: str) -> Any:
    """
    Restituisce l'embedding per una singola stringa usando Settings.embed_model.
    """
    model = Settings.embed_model
    return _get_embedding_from_model(model, text)

def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError("Per leggere i PDF installa PyPDF2: pip install PyPDF2") from e
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            # ignora pagine problematiche
            continue
    return "\n".join(pages)

def embed_file(file_path: str) -> Dict[str, Any]:
    """
    Genera l'embedding per un singolo file.
    Supporta txt, md, pdf. Restituisce un dict con 'path', 'text', 'embedding'.
    """
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File non trovato: {file_path}")
    suffix = p.suffix.lower()
    if suffix in {".txt", ".md", ".py", ".csv", ".json"}:
        text = _read_text_file(p)
    elif suffix == ".pdf":
        text = _read_pdf(p)
    else:
        # tenta come testo altrimenti errore
        try:
            text = _read_text_file(p)
        except Exception:
            raise RuntimeError(f"Estensione non supportata e non è stato possibile leggere come testo: {file_path}")
    embedding = embed_text(text)
    return {"path": str(p), "text": text, "embedding": embedding}

def embed_folder(folder_path: str, recurse: bool = False, exts: List[str] = None) -> Dict[str, Any]:
    """
    Scansiona la cartella e genera embedding per tutti i file supportati.
    - folder_path: percorso della cartella
    - recurse: se True scansiona ricorsivamente
    - exts: lista opzionale di estensioni da includere (es. ['.txt', '.pdf'])
    Restituisce un dizionario {path: embedding_result}
    """
    p = Path(folder_path)
    if not p.exists() or not p.is_dir():
        raise NotADirectoryError(f"Cartella non trovata: {folder_path}")
    if exts:
        exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    else:
        exts = {".txt", ".md", ".pdf", ".py", ".csv", ".json"}

    results = {}
    if recurse:
        files = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in exts]
    else:
        files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts]

    for f in files:
        try:
            res = embed_file(str(f))
            results[res["path"]] = res
        except Exception as e:
            # non interrompere l'intero processo per un singolo file problematico
            results[str(f)] = {"error": str(e)}
    return results