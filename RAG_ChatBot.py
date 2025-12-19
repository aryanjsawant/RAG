# RAG_ChatBot.py

import os
import hashlib
import streamlit as st

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader
)

# ================================
# CONFIG
# ================================
MATERIALS_DIR = "./materials"
FAISS_INDEX_PATH = "./faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

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
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
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

    # ---- FAISS ----
    if os.path.exists(FAISS_INDEX_PATH):
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        vector_store = FAISS.from_documents(all_docs, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)

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
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            device_map="auto",
            torch_dtype="auto"
        )

        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=True
        )

        return HuggingFacePipeline(pipeline=text_gen_pipeline)

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
