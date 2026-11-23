
from rag.retrieval import retrieve_documents, format_context
from llm import generate_answer


def ask_chatbot(question):

    question = str(question).strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": []
        }

    try:
        # 1. Retrieve relevant medical information
        documents = retrieve_documents(
            question,
            top_k=3
        )

        if not documents:
            return {
                "answer": (
                    "I could not find relevant information "
                    "in the FractureAI medical knowledge base. "
                    "Please consult a qualified healthcare professional."
                ),
                "sources": []
            }

        # 2. Prepare retrieved context
        context = format_context(documents)

        # 3. Create Qwen prompt
        prompt = f"""
You are FractureAI, an educational bone health assistant.

Answer using only the medical information in the context.

IMPORTANT RULES:
- Use simple, clear English.
- Do not invent medical facts.
- Do not repeat the same warning sign.
- Do not claim to diagnose a fracture.
- If information is missing, say so.
- For severe injuries, recommend urgent medical assessment.
- Never recommend moving a severely injured limb unnecessarily.

MEDICAL CONTEXT:
{context}

USER QUESTION:
{question}

Give a clear and helpful answer:
"""

        # 4. Generate answer using local Qwen
        answer = generate_answer(prompt)

        # 5. Collect source filenames
        sources = list(set(
            document["source"]
            for document in documents
        ))

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as error:

        print("FRACTUREAI CHATBOT ERROR:", error)

        return {
            "answer": (
                "Sorry, the FractureAI chatbot encountered "
                "an error. Please try again."
            ),
            "sources": []
        }
