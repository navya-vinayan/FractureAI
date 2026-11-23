
from rag.retrieval import retrieve_documents, format_context
from llm import generate_answer


def ask_fractureai(question):

    # Step 1: Retrieve relevant medical information
    documents = retrieve_documents(question, top_k=3)

    if not documents:
        return "I could not find relevant information in the medical knowledge base."

    # Step 2: Prepare context
    context = format_context(documents)

    # Step 3: Create prompt for Qwen
    prompt = f"""
You are FractureAI, an educational bone health assistant.

Answer the user's question using ONLY the medical information
provided in the context below.

Rules:
- Use simple, clear English.
- Do not invent medical facts.
- If the context does not contain the answer, say so.
- Never claim to diagnose a patient.
- Recommend professional medical assessment when appropriate.

MEDICAL CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    # Step 4: Generate answer using Qwen
    answer = generate_answer(prompt)

    # Step 5: Return answer and sources
    sources = list(set(
        document["source"] for document in documents
    ))

    return {
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":

    question = input("Ask FractureAI: ")

    result = ask_fractureai(question)

    print("\nFRACTUREAI ANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for source in result["sources"]:
        print("-", source)