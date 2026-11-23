
from sentence_transformers import SentenceTransformer


# Load the embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = None


def load_embedding_model():

    global embedding_model

    if embedding_model is None:
        print("Loading embedding model...")

        embedding_model = SentenceTransformer(MODEL_NAME)

        print("Embedding model loaded successfully!")

    return embedding_model


def generate_embeddings(texts):

    model = load_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings.tolist()


def generate_query_embedding(question):

    model = load_embedding_model()

    embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    return embedding.tolist()
