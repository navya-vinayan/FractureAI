
from .vector_store import get_collection
from .embeddings import generate_query_embedding


def retrieve_documents(question, top_k=3):

    collection = get_collection()

    # Convert the user's question into an embedding
    query_embedding = generate_query_embedding(question)

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_documents = []

    if results["documents"]:

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for document, metadata, distance in zip(
            documents, metadatas, distances
        ):

            retrieved_documents.append({
                "text": document,
                "source": metadata["source"],
                "distance": distance
            })

    return retrieved_documents


def format_context(retrieved_documents):

    context = ""

    for document in retrieved_documents:

        context += (
            f"\nSource: {document['source']}\n"
            f"{document['text']}\n"
        )

    return context.strip()
