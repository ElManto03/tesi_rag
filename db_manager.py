import os
import json
from abc import ABC, abstractmethod
import psycopg
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from models import User
from config import settings

# Inizializziamo l'engine una sola volta a livello di modulo per gestire meglio
# il pool di connessioni, invece di ricrearlo per ogni file salvato.
db_engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI), future=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

def ottenere_utente_da_db(email: str):
    """
    Versione isolata: apre e chiude la sessione in autonomia.
    Nessun impatto sul resto del progetto.
    """
    db = SessionLocal() # Apre la connessione a Postgres
    try:
        utente = db.query(User).filter(User.email == email).first()
        return utente
    finally:
        db.close()


class VectorStoreAdapter(ABC):
    @abstractmethod
    def save_document(self, file_name: str, file_url: str, access_level: str, total_pages: int, chunks: list, file_hash: str = None, markdown_hash: str = None, created_at: str = None, expiry_date: str = None, db_role: str = 'user') -> int:
        pass

class PostgresStoreAdapter(VectorStoreAdapter):
    def __init__(self, engine):
        self.engine = engine

    def save_document(self, file_name, file_url, access_level, total_pages, chunks, file_hash=None, md_hash=None, created_at=None, expiry_date=None, db_role='user'):
        if not access_level:
            raise ValueError("Access level mancante; rifiuto il salvataggio del documento.")

        try:
            with self.engine.begin() as conn:
                # Configura le variabili di sessione per RLS (Row Level Security)
                # Il parametro 'true' in set_config rende l'impostazione locale alla transazione
                conn.execute(text("SELECT set_config('app.current_role', :role, true)"), {"role": db_role})
                conn.execute(text("SELECT set_config('app.current_user_level', :level, true)"), {"level": access_level})

                # Controllo se un documento con lo stesso hash esiste già per sostituirlo
                if file_hash:
                    existing_id = conn.execute(
                        text("SELECT id FROM documents WHERE file_hash = :fh"), {"fh": file_hash}
                    ).scalar()

                    if existing_id:
                        # Rimuoviamo i vecchi chunk e il documento per permettere la sostituzione integrale
                        conn.execute(text("DELETE FROM document_chunks WHERE parent_doc_id = :id"), {"id": existing_id})
                        conn.execute(text("DELETE FROM documents WHERE id = :id"), {"id": existing_id})
                        print(f"🔄 Documento con hash {file_hash[:10]}... già presente: vecchia versione rimossa per sostituzione.")

                document_id = conn.execute(
                    text("INSERT INTO documents (file_name, upload_date, file_url, access_level, total_pages, file_hash, md_hash, expiry_date) "
                         "VALUES (:file_name, :upload_date, :file_url, :access_level, :total_pages, :file_hash, :md_hash, :expiry_date) RETURNING id"),
                    {
                        "file_name": file_name, 
                        "upload_date": created_at,
                        "file_url": file_url, 
                        "access_level": access_level, 
                        "total_pages": total_pages,
                        "file_hash": file_hash,
                        "md_hash": md_hash,
                        "expiry_date": expiry_date
                    }
                ).scalar_one()

                chunk_rows = [
                    {"parent_doc_id": document_id, "content": chunk["text"], "chunk_index": chunk["chunk_id"],
                     "encrypted_content": chunk.get("encrypted_text"),
                     "page_number": chunk.get("page_number"), "embedding": chunk.get("embedding"),
                     "offset_from_start": chunk.get("offset_from_start"), "access_level": access_level} for chunk in chunks
                ]
                if chunk_rows:
                    # sql = "INSERT INTO document_chunks (parent_doc_id, content, encrypted_content, chunk_index, page_number, embedding, offset_from_start) VALUES (:parent_doc_id, :content, :encrypted_content, :chunk_index, :page_number, :embedding, :offset_from_start)"
                    conn.execute(text("INSERT INTO document_chunks (parent_doc_id, content, encrypted_content, chunk_index, page_number, embedding, offset_from_start, access_level) "
                                      "VALUES (:parent_doc_id, :content, :encrypted_content, :chunk_index, :page_number, :embedding, :offset_from_start, :access_level)"), chunk_rows)
            print(f"✅ Documento salvato su Postgres con id {document_id}.")
            return document_id
        except SQLAlchemyError as exc:
            print(f"❌ Errore Postgres: {exc}")
            raise

class QdrantStoreAdapter(VectorStoreAdapter):
    def __init__(self, url=None, api_key=None):
        self.url = url
        # Qui inizializzeresti qdrant_client.QdrantClient

    def save_document(self, file_name, file_url, access_level, total_pages, chunks, file_hash=None, markdown_hash=None, created_at=None, expiry_date=None):
        print(f"🚀 [Qdrant] Simulazione salvataggio per {file_name}. (Da implementare con qdrant-client)")
        return 0

def create_db_and_tables() -> None:
    """
    Crea le tabelle necessarie nel database se non esistono.
    Nota: Assicurati che le tabelle 'documents' e 'document_chunks' siano definite
    in un modello SQLModel o tramite script SQL esterni.
    Questa funzione è un placeholder per l'inizializzazione del DB.
    """
    # Esempio di creazione tabelle se non esistono (adatta al tuo schema SQLModel)
    # from sqlmodel import SQLModel
    # SQLModel.metadata.create_all(db_engine)
    print("Database tables creation (if not exists) handled by SQLModel or external scripts.")
    pass


def save_document_and_chunks_to_db(file_name, file_url, access_level, total_pages, chunks, adapter: VectorStoreAdapter = None, db_role: str = 'user', **kwargs):
    if adapter is None:
        adapter = PostgresStoreAdapter(db_engine)
    return adapter.save_document(file_name, file_url, access_level, total_pages, chunks, db_role=db_role, **kwargs)

def salva_audit_log_bg(user_role: str, security_blocked: bool, log_data_dict: dict):
    """
    Salva il report dell'audit trail su database forzando il ruolo developer 
    per superare i vincoli di sicurezza RLS.
    """
    query = text("""
        INSERT INTO audit_logs (user_role, security_blocked, log_data)
        VALUES (:user_role, :security_blocked, :log_data);
    """)
    
    try:
        with db_engine.connect() as conn:
            # Forziamo il ruolo developer per la sessione corrente di scrittura log
            #conn.execute(text("SET app.current_role = 'developer';"))
            conn.execute(text("SELECT set_config('app.current_role', 'developer', false);"))
            
            conn.execute(query, {
                "user_role": user_role,
                "security_blocked": security_blocked,
                "log_data": json.dumps(log_data_dict)
            })
            conn.commit()
            print("[AUDIT DB] Log di audit salvato con successo.")
    except Exception as e:
        print(f"[AUDIT DB] ERRORE CRITICO durante il salvataggio del log: {e}")

def salva_ingestion_log(user_role: str, file_name: str, file_hash: str, ingestion_data_dict: dict):
    """
    Salva i metadati di ingestion sul database forzando il ruolo developer.
    """
    query = text("""
        INSERT INTO ingestion_logs (user_role, file_name, file_hash_sha256, ingestion_data)
        VALUES (:user_role, :file_name, :file_hash, :ingestion_data);
    """)
    try:
        with db_engine.connect() as conn:
            #conn.execute(text("SET app.current_role = 'developer';"))
            conn.execute(text("SELECT set_config('app.current_role', 'developer', false);"))
            conn.execute(query, {
                "user_role": user_role,
                "file_name": file_name,
                "file_hash": file_hash,
                "ingestion_data": json.dumps(ingestion_data_dict)
            })
            conn.commit()
            print(f"[INGESTION DB] Log salvato con successo per il file: {file_name}")
    except Exception as e:
        print(f"[INGESTION DB] ERRORE durante il salvataggio del log di ingestion: {e}")
