# RAG_ChatBot.py

import os
import hashlib
import streamlit as st
from typing import List

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_chroma import Chroma

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader
)

# ================================
# CONFIG
# ================================
OLLAMA_BASE_URL = "http://localhost:11434"
CHROMA_DIR = "./chroma_db"
MATERIALS_DIR = "./materials"

# ================================
# UTILS
# ================================
def file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def format_docs(docs: List[Document]) -> str:
    formatted = []
    for d in docs:
        source = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page", "N/A")
        formatted.append(
            f"[{source}, page {page}]\n{d.page_content}"
        )
    return "\n\n".join(formatted)

# ================================
# VECTOR STORE (CHROMA)
# ================================
@st.cache_resource(show_spinner="Indexing documents...")
def load_vector_store():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_BASE_URL
    )

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="electrical_standards"
    )

    existing = vector_store.get(include=["ids"])
    existing_ids = set(existing["ids"])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150
    )

    new_docs = []
    new_ids = []

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
                "hash": fhash
            })

            doc_id = f"{filename}_{fhash}_{i}"

            if doc_id not in existing_ids:
                new_docs.append(doc)
                new_ids.append(doc_id)

    if new_docs:
        vector_store.add_documents(new_docs, ids=new_ids)
        vector_store.persist()

    return vector_store

# ================================
# CHATBOT
# ================================
class ChatBot:
    def __init__(self):
        self.vector_store = load_vector_store()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        self.llm = self._load_llm()
        self.chain = self._build_chain()

    def _load_llm(self):
        return ChatOllama(
            model="llama3",
            temperature=0,
            base_url=OLLAMA_BASE_URL
        )

    def _build_chain(self):
        template = """
You are an Electrical Standards Assistant.

Rules:
- Answer ONLY using the provided documents.
- Do NOT use outside knowledge.
- If the answer is not found, say exactly:
  "I don't know based on the provided standards."
- Cite sources inline as: (Document, page)

Context:
{context}

Question:
{question}

Answer (maximum 2 sentences):
"""

        prompt = PromptTemplate.from_template(template)

        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain
