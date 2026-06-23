from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, Date, ForeignKey, CHAR
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

# La classe Base da cui erediteranno tutti i nostri modelli
Base = declarative_base()

# =========================================================================
# 1. MODELLO: UTENTI (users)
# =========================================================================
class User(Base):
    """
    Tabella di anagrafica utenti autenticati tramite Google SSO.
    Non memorizza credenziali o hash di password.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    google_id = Column(String(255), nullable=False, unique=True)
    user_role = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


# =========================================================================
# 2. MODELLO: DOCUMENTI GENITORI (documents)
# =========================================================================
class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default="gen_random_uuid()")
    file_name = Column(Text, nullable=False)
    file_url = Column(Text, nullable=False)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    access_level = Column(Text, nullable=False)
    total_pages = Column(Integer, nullable=True)
    file_hash = Column(CHAR(256), nullable=False)
    md_hash = Column(CHAR(256), nullable=False)
    expiry_date = Column(Date, nullable=True)

    # Relazione virtuale verso i figli (comoda per fare document.chunks in Python)
    chunks = relationship("DocumentChunk", back_populates="parent_document", cascade="all, delete-orphan")


# =========================================================================
# 3. MODELLO: CHUNK FIGLI (document_chunks)
# =========================================================================
class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default="gen_random_uuid()")
    parent_doc_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    encrypted_content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=True)
    page_number = Column(Integer, nullable=True)
    
    # Nota per la tesi: Gestiamo il tipo VECTOR di pgvector come tipo generico se non usi l'estensione pgvector in Python,
    # altrimenti puoi mappare come Text o usare la libreria pgvector. Per ora lo commentiamo o definiamo come Text/NullType.
    # embedding = Column(Text) 
    
    offset_from_start = Column(Integer, nullable=True)
    access_level = Column(Text, nullable=False)

    # Relazione inversa verso il genitore
    parent_document = relationship("Document", back_populates="chunks")


# =========================================================================
# 4. MODELLO: LOG DI AUDIT CHAT (audit_logs)
# =========================================================================
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)
    user_role = Column(String(50), nullable=True)
    security_blocked = Column(Boolean, default=False)
    log_data = Column(JSONB, nullable=True)


# =========================================================================
# 5. MODELLO: LOG DI INGESTION (ingestion_logs)
# =========================================================================
class IngestionLog(Base):
    __tablename__ = "ingestion_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)
    user_role = Column(String(50), nullable=True)
    file_name = Column(String(255), nullable=True)
    file_hash_sha256 = Column(String(64), nullable=True)
    ingestion_data = Column(JSONB, nullable=True)