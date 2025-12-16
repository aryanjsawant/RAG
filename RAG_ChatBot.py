# RAG_ChatBot.py

import os
import hashlib
import streamlit as st

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader
)

# ================================
# CONFIG
# ================================
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"  # fallback for local dev
)

MATERIALS_DIR = "./materials"

# ================================
# UTILS
# ================================
def file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

# ================================
# VECTOR STORE (CACHED)
# ================================
@st.cache_resource(show_spinner="Indexing documents...")
def load_vector_store():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_BASE_URL
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150
    )

    all_docs = []

    for filename in os.listdir(MATERIALS_DIR):
        filepath = os.path.join(MATERIALS_DIR, filename)
        fhash = file_hash(filepath)

        if filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(filepath)
        else:
            continue

        raw_docs = loader.load()
        docs = splitter.split_documents(raw_docs)

        for i, doc in enumerate(docs):
            doc.metadata.update({
                "source": filename,
                "page": doc.metadata.get("page", "N/A"),
                "hash": fhash,
                "chunk": i
            })
            all_docs.append(doc)

    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(all_docs)

    return vector_store

# ================================
# CHATBOT
# ================================
class ChatBot:
    def __init__(self):
        self.vector_store = load_vector_store()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        self.llm = self._load_llm()
        self.rag_chain = self._build_chain()

    def _load_llm(self):
        return ChatOllama(
            model="llama3",
            temperature=0.2,
            base_url=OLLAMA_BASE_URL
        )

    def _build_chain(self):
        template = """
You are an Electrical Standards Assistant.

Rules:
- Answer ONLY from the provided documents.
- If the answer is not present, say:
  "I don't know based on the provided standards."
- Be concise and factual.

Context:
{context}

Question:
{question}

Answer (max 2 sentences):
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False
        )

# ================================
# LOCAL TEST
# ================================
if __name__ == "__main__":
    bot = ChatBot()
    print(bot.rag_chain.invoke("What is the scope of the documents?"))
