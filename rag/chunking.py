
import os


def load_documents(folder_path):
    documents = []

    for filename in os.listdir(folder_path):

        if filename.endswith(".txt"):

            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            if text.strip():
                documents.append({
                    "text": text,
                    "source": filename
                })

    return documents


def split_text(text, chunk_size=700, overlap=100):

    text = " ".join(text.split())

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size

        if end < len(text):
            space_position = text.rfind(" ", start, end)

            if space_position > start:
                end = space_position

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def create_document_chunks(folder_path):

    documents = load_documents(folder_path)

    all_chunks = []

    for document in documents:

        chunks = split_text(document["text"])

        for index, chunk in enumerate(chunks):

            all_chunks.append({
                "text": chunk,
                "source": document["source"],
                "chunk_id": index
            })

    return all_chunks
