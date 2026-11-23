
import chromadb
from pathlib import Path

from .embeddings import generate_embeddings


# ChromaDB storage location
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "fractureai_knowledge"


# Create ChromaDB client
client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)


# Create or access collection
def get_collection():

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


# Store document chunks and embeddings
def store_chunks(chunks):

    if not chunks:
        print("No document chunks found!")
        return

    collection = get_collection()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = generate_embeddings(texts)

    ids = [
        f'{chunk["source"]}_{chunk["chunk_id"]}'
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"]
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Successfully stored {len(chunks)} chunks in ChromaDB!")


# Check stored document count
def get_document_count():

    collection = get_collection()

    return collection.count()
