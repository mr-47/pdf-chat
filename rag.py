from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import numpy as np
import os
import sys
import tempfile

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MODEL_NAME = "all-MiniLM-L6-v2"

_SILENCE_INSTALLED = False


def _silence_unused_vision_import_errors():
    """Hide the ModuleNotFoundError for torchvision.

    transformers 5.x ships roughly a hundred vision model modules that import
    torchvision at module level with no backend guard, so any of them raises
    when torchvision is not installed. This project only does text embedding
    and never loads a vision model, so the traceback is pure noise.

    The trigger is not our code. Streamlit's development file watcher walks
    sys.modules and calls hasattr(module, "__path__") on every entry. Because
    transformers uses a lazy _LazyModule, that hasattr really does import the
    submodule, which then hits the missing torchvision. Streamlit catches it
    in a bare except and logs it as a warning.

    Any log record can reach the terminal through several sinks, and which one
    is live depends on the order libraries configure logging. Handlers are
    therefore filtered wherever they are: the existing ones are swept, and
    Logger.addHandler is wrapped so handlers created later are filtered too.
    Only records mentioning torchvision are dropped; every other message,
    warning and error still passes through untouched.
    """
    global _SILENCE_INSTALLED
    if _SILENCE_INSTALLED:
        return
    _SILENCE_INSTALLED = True

    import logging

    class _Filter(logging.Filter):
        def filter(self, record):
            try:
                text = record.getMessage()
            except Exception:
                text = str(record.msg)
            if record.exc_info and record.exc_info[1] is not None:
                text = f"{text} {record.exc_info[1]}"
            return "torchvision" not in text

    _FILTER = _Filter()

    def attach(handler):
        if handler is not None and _FILTER not in handler.filters:
            handler.addFilter(_FILTER)

    def sweep():
        for logger in [logging.getLogger(), *logging.Logger.manager.loggerDict.values()]:
            if isinstance(logger, logging.Logger):
                for handler in list(logger.handlers):
                    attach(handler)
        attach(logging.lastResort)

    sweep()

    if not getattr(logging.Logger.addHandler, "_pdfchat_wrapped", False):
        original_add_handler = logging.Logger.addHandler

        def add_handler(self, handler):
            original_add_handler(self, handler)
            attach(handler)

        add_handler._pdfchat_wrapped = True
        logging.Logger.addHandler = add_handler


_silence_unused_vision_import_errors()


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Load the embedding model, preferring the local copy.

    Without this, every run contacts huggingface.co with HEAD requests to
    check whether a newer revision of the model exists, which is slow, leaks
    usage data and fails on a machine with no network. The model is pinned by
    cache revision, so staying offline changes nothing about the output.

    Set PDFCHAT_ALLOW_DOWNLOAD=1 to force a network check, for example after
    clearing the cache to pick up a different model version.
    """
    _silence_unused_vision_import_errors()

    if os.environ.get("PDFCHAT_ALLOW_DOWNLOAD"):
        return SentenceTransformer(model_name)

    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        print(
            f"Embedding model not cached yet, downloading {model_name} "
            "from Hugging Face (~90 MB, this happens once).",
            file=sys.stderr,
        )
        return SentenceTransformer(model_name)


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = _load_model(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


class VectorStore:
    """Exhaustive squared-L2 search over every chunk.

    Replaces FAISS. A typical document yields a few hundred chunks, so a full
    scan is instant and avoids a ~70MB native dependency. Scores are squared
    L2 distances, the same convention FAISS uses for an L2 index.
    """

    def __init__(self, chunks, vectors):
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def similarity_search_with_score(self, query, k: int = 4):
        vector = np.asarray(self.embed_query(query), dtype=np.float32)
        distances = ((self.vectors - vector) ** 2).sum(axis=1)
        order = np.argsort(distances)[:k]
        return [(self.chunks[i], float(distances[i])) for i in order]

    def similarity_search(self, query, k: int = 4):
        return [doc for doc, _ in self.similarity_search_with_score(query, k)]

    def embed_query(self, text):
        return _load_model(MODEL_NAME).encode([text])[0]


def load_pdf_from_path(pdf_path: str) -> list[Document]:
    reader = PdfReader(pdf_path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": pdf_path, "page": number},
        )
        for number, page in enumerate(reader.pages)
    ]


def load_pdf_from_bytes(pdf_bytes: bytes) -> list[Document]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        return load_pdf_from_path(tmp_path)
    finally:
        os.remove(tmp_path)


def index_documents(docs: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError(
            "No text could be extracted. The PDF is probably scanned "
            "images and needs OCR."
        )

    vectors = _load_model(MODEL_NAME).encode(
        [chunk.page_content for chunk in chunks]
    )
    return VectorStore(chunks, vectors), chunks


def build_index_from_path(pdf_path: str):
    return index_documents(load_pdf_from_path(pdf_path))


def build_index_from_bytes(pdf_bytes: bytes):
    return index_documents(load_pdf_from_bytes(pdf_bytes))


def index_pages(chunks: list[Document]) -> list[int]:
    return sorted({chunk.metadata.get("page", 0) for chunk in chunks})


def retrieve_context(vectorstore, query: str, k: int = 4):
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)
    return context, docs


def retrieve_matches(vectorstore, query: str, k: int = 4):
    matches = vectorstore.similarity_search_with_score(query, k=k)
    return sorted(matches, key=lambda pair: pair[1])
