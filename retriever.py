import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

VECTORSTORE_PATH = "vectorstore_index"

def load_docs(path="./data"):
    docs = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                content = f.read()
                docs.append(Document(
                    page_content=content,
                    metadata={"source": filename}
                ))
    if not docs:
        raise ValueError("No .txt documents found in /data.")
    return docs

def build_or_load_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(VECTORSTORE_PATH):
        return FAISS.load_local(
            folder_path=VECTORSTORE_PATH,
            embeddings=embedding,
            allow_dangerous_deserialization=True
        )

    docs = load_docs()
    vectorstore = FAISS.from_documents(docs, embedding)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore

vectorstore = build_or_load_vectorstore()