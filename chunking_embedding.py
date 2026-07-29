import os
import re
from abc import ABC, abstractmethod
import pysbd
from dotenv import load_dotenv, set_key
from llama_index.core.node_parser import SemanticSplitterNodeParser, MarkdownNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Document, Settings
from cryptography.fernet import Fernet

# Carica le variabili d'ambiente dal file .env se esistente
load_dotenv()

def get_or_create_encryption_key():
    key_var = "DOC_ENCRYPTION_KEY"
    env_path = ".env"
    key = os.getenv(key_var)
    
    if not key:
        # Genera una nuova chiave e la converte in stringa per il salvataggio
        key = Fernet.generate_key().decode()
        # Salva la chiave nel file .env
        set_key(env_path, key_var, key)
        print(f"🔑 Nuova chiave di cifratura generata e salvata in {env_path}")
    return key

ENCRYPTION_KEY = get_or_create_encryption_key()
cipher_suite = Fernet(ENCRYPTION_KEY.encode())

scuola_info = {"indirizzo": "Via Ada Negri, 34 - 47923 Rimini (RN)", "tel": "(+39) 0541 384159",  "cf": "82007870403", "web": "itsrimini.edu.it", "mail": "RNTF010004@istruzione.it", "pec": "RNTF010004@pec.istruzione.it"}

class EmbeddingProvider(ABC):
    @abstractmethod
    def get_model(self):
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: list) -> list:
        pass

class OllamaEmbeddingAdapter(EmbeddingProvider):
    def __init__(self, model_name="qwen3-embedding:8b"):
        self.model_name = model_name
        self.embed_model = OllamaEmbedding(
            model_name=self.model_name, 
            base_url=os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
        )

    def get_model(self):
        return self.embed_model

    def get_embeddings(self, texts):
        return self.embed_model.get_text_embedding_batch(texts)

class VLLMEmbeddingAdapter(EmbeddingProvider):
    def __init__(self, model_name, api_url):
        self.model_name = model_name
        self.api_url = api_url

    def get_model(self): return None # vLLM si usa solitamente via API REST diretta
    def get_embeddings(self, texts):
        print(f"🚀 [vLLM] Generazione embedding per {len(texts)} testi tramite {self.api_url}")
        return [[] for _ in texts] # Placeholder


class LegalSegmenter:
    """
    Segmentatore di frasi personalizzato per testi legali/amministrativi italiani,
    con gestione degli acronimi per evitare split errati.
    """
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

def is_markdown_table(text: str) -> bool:
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("|") and cleaned.endswith("|"):
            stripped = cleaned.replace("|", "").replace("-", "").replace(":", "").replace(" ", "")
            if len(stripped) == 0 and len(cleaned) > 2:
                return True
    return False

def pre_split_by_headings(nodes):
    """
    Se un nodo contiene al suo interno dei titoli principali (# o ##), 
    lo divide in sotto-nodi in modo che ogni titolo inizi un nuovo blocco.
    """
    split_nodes = []
    # Regex per catturare titoli di livello 1 e 2 (es. # TITOLO o ## TAB. 1)
    heading_pattern = re.compile(r'^(##? \s*.+)$', re.MULTILINE)
    
    for node in nodes:
        text = node.text
        parts = heading_pattern.split(text)
        
        # Se non ci sono titoli intermedi, tieni il nodo così com'è
        if len(parts) <= 1:
            split_nodes.append(node)
            continue
            
        current_text = ""
        # Il primo elemento è il testo prima del primo titolo (se presente)
        if parts[0].strip():
            # Creiamo un duplicato del nodo con il testo parziale
            from llama_index.core import Document # o la classe che usi per i nodi
            split_nodes.append(Document(text=parts[0].strip(), metadata=node.metadata))
            
        # Ricostruiamo le coppie (Titolo, Contenuto successivo)
        for i in range(1, len(parts), 2):
            heading = parts[i]
            content = parts[i+1] if (i+1) < len(parts) else ""
            combined = (heading + "\n" + content).strip()
            if combined:
                from llama_index.core import Document
                split_nodes.append(Document(text=combined, metadata=node.metadata))
                
    return split_nodes

def chunk_splitter(docs, splitter, md_parser):
    # 1. Parsing iniziale
    raw_nodes = md_parser.get_nodes_from_documents(docs)
    
    # 2. Pre-segmentazione sui titoli per evitare che macro-sezioni rimangano unite
    initial_nodes = pre_split_by_headings(raw_nodes)
    
    nodes = []
    
    if len(initial_nodes) > 0:
        MIN_CHUNK_SIZE, MAX_CHUNK_SIZE = 200, 2000
        buffer_text, buffer_metadata = "", {}
        
        for node in initial_nodes:
            current_text = node.text.strip()
            current_metadata = node.metadata
            
            is_current_table = is_markdown_table(current_text)
            starts_with_heading = current_text.startswith("#")
            
            # FLUSH PREVENTIVO:
            # Se nel buffer c'è qualcosa e il nodo corrente è un titolo o una tabella,
            # salviamo immediatamente il buffer come chunk a sé stante.
            if buffer_text and (starts_with_heading or is_current_table):
                from llama_index.core import Document
                nodes.append(Document(text=buffer_text, metadata=buffer_metadata))
                buffer_text, buffer_metadata = "", {}
            
            if buffer_text:
                combined_text = buffer_text + "\n\n" + current_text
                combined_metadata = {**buffer_metadata, **current_metadata}
            else:
                combined_text = current_text
                combined_metadata = current_metadata
            
            is_combined_table = is_markdown_table(combined_text)
            
            # CONTROLLO ANTI-TITOLO ORFANO
            ends_with_heading = False
            lines = combined_text.splitlines()
            if lines:
                last_line = lines[-1].strip()
                if last_line.startswith("###") or last_line.startswith("##"):
                    ends_with_heading = True

            # DECISIONE: ACCUMULARE O EMETTERE
            if (len(combined_text) < MIN_CHUNK_SIZE or ends_with_heading) and not is_combined_table:
                buffer_text, buffer_metadata = combined_text, combined_metadata
            else:
                from llama_index.core import Document
                new_doc = Document(text=combined_text, metadata=combined_metadata)
                
                if len(combined_text) > MAX_CHUNK_SIZE:   
                    splitted = splitter.get_nodes_from_documents([new_doc])
                    nodes.extend(splitted if all(len(n.text) >= MIN_CHUNK_SIZE for n in splitted) else [new_doc])
                else: 
                    nodes.append(new_doc)
                
                buffer_text, buffer_metadata = "", {}
                
        # Residuo finale
        if buffer_text:
            if nodes: 
                from llama_index.core import Document
                nodes[-1] = Document(text=nodes[-1].text + "\n\n" + buffer_text, metadata={**nodes[-1].metadata, **buffer_metadata})
            else:
                from llama_index.core import Document
                nodes.append(Document(text=buffer_text, metadata=buffer_metadata))
    else: 
        nodes = initial_nodes
        
    return nodes


def custom_sentence_splitter(text):
    return LegalSegmenter().split_sentences(text)


def get_node_page_number(node):
    """Estrae il numero di pagina dai metadati del nodo."""
    metadata = getattr(node, "metadata", {}) or {}
    page = metadata.get("page_number") or metadata.get("page")
    if page is not None:
        try:
            return int(page)
        except (ValueError, TypeError):
            return None
    return None


def run_semantic_chunking(text, metadata=None, embed_provider=None):
    """
    Esegue il chunking semantico e l'embedding di un testo.
    """
    if embed_provider is None:
        embed_provider = OllamaEmbeddingAdapter()
    md_parser = MarkdownNodeParser()
    splitter = SemanticSplitterNodeParser(
        buffer_size=3, breakpoint_percentile_threshold=95, 
        embed_model=embed_provider.get_model(), sentence_splitter=custom_sentence_splitter
    )

    # Suddividiamo il testo per pagine e rimuoviamo i marcatori di splitting PRIMA del parsing.
    # Questo permette al MarkdownNodeParser e al SemanticSplitter di lavorare su testo pulito.
    page_pattern = r'\{(\d+)\}-+'
    matches = list(re.finditer(page_pattern, text))
    docs = []
    
    excluded_keys = ["cf", "mail", "pec", "indirizzo", "tel", "web", "source_path", "access_level", "page_number"]

    if not matches:
        docs = [Document(text=text, metadata={"page_number": 1}, 
                         excluded_embed_metadata_keys=excluded_keys)]
    else:
        # Gestione testo prima del primo tag (solitamente vuoto se il file inizia con {0}---)
        if matches[0].start() > 0:
            docs.append(Document(text=text[:matches[0].start()].strip(), 
                                 metadata={"page_number": 1}, 
                                 excluded_embed_metadata_keys=excluded_keys))
        
        for i, match in enumerate(matches):
            page_num = int(match.group(1)) + 1
            start = match.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            page_content = text[start:end].strip()
            if page_content:
                docs.append(Document(text=page_content, 
                                    metadata={"page_number": page_num}, 
                                    excluded_embed_metadata_keys=excluded_keys))

    nodes = chunk_splitter(docs, splitter, md_parser)

    chunk_texts = [node.get_content() for node in nodes]
    embeddings = embed_provider.get_embeddings(chunk_texts)

    results = []
    current_search_pos = 0
    for i, (node, embedding) in enumerate(zip(nodes, embeddings)):
        content = node.get_content()
        # Cerchiamo la posizione del chunk nel testo originale a partire dall'ultimo punto trovato
        start_char_index = text.find(content, current_search_pos)
        
        if start_char_index != -1:
            # L'offset del prossimo chunk deve partire dopo la fine di quello attuale
            current_search_pos = start_char_index + len(content)

        encrypted_text = cipher_suite.encrypt(content.encode()).decode()

        results.append({
            "chunk_id": i, 
            "text": content, 
            "encrypted_text": encrypted_text,
            "page_number": get_node_page_number(node), 
            "embedding": embedding,
            "offset": start_char_index if start_char_index != -1 else None
        })
    return results