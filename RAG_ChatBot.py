import os
import hashlib
import streamlit as st
import re
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ================================
# CONFIG
# ================================
MATERIALS_DIR = "./materials"
FAISS_INDEX_PATH = "./faiss_index"

# Ensure you use these specific paths on your server
EMBEDDING_MODEL = "/home/aryan/models/all-mpnet"
LLM_MODEL = "/home/aryan/models/mistral"

# ================================
# UTILS
# ================================
def file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def get_3gpp_release(filename):
    """Maps 3GPP letter codes (-f, -g, etc.) to Release numbers."""
    fn = filename.lower()
    if "-f" in fn: return "Release 15"
    if "-g" in fn: return "Release 16"
    if "-h" in fn: return "Release 17"
    if "-i" in fn: return "Release 18"
    if "-j" in fn: return "Release 19"
    if "-k" in fn: return "Release 20"
    
    # Fallback for custom named files like TS_23501_Rel15
    rel_match = re.search(r'Rel(\d+)', filename, re.IGNORECASE)
    if rel_match:
        return f"Release {rel_match.group(1)}"
    
    return "Unknown Release"

def format_docs(docs):
    """Convert retrieved Documents to a context string with explicit Source/Release headers."""
    formatted_context = ""
    for doc in docs:
        source = doc.metadata.get("source", "Unknown File")
        release = doc.metadata.get("release", "Unknown Release")
        formatted_context += f"\n--- DOCUMENT: {source} | VERSION: {release} ---\n{doc.page_content}\n"
    return formatted_context

# ================================
# VECTOR STORE (CACHED)
# ================================
@st.cache_resource(show_spinner="Indexing documents with Release mapping...")
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_docs = []

    # HARD RESET: Deletes old index to ensure new Release mapping is applied
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)

    if not os.path.exists(MATERIALS_DIR):
        raise FileNotFoundError(f"{MATERIALS_DIR} directory not found")

    for filename in os.listdir(MATERIALS_DIR):
        filepath = os.path.join(MATERIALS_DIR, filename)
        fhash = file_hash(filepath)
        release_tag = get_3gpp_release(filename)

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

        for doc in docs:
            # INJECTION: Prepend the release to the text chunk itself
            # This makes the version "visible" to the LLM's attention mechanism.
            doc.page_content = f"[[CONTEXT: This information is strictly from 3GPP {release_tag}]]\n{doc.page_content}"
            
            doc.metadata.update({
                "source": filename,
                "release": release_tag,
                "hash": fhash,
            })
            all_docs.append(doc)

    vector_store = FAISS.from_documents(all_docs, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    return vector_store

# ================================
# CHATBOT
# ================================
class ChatBot:
    def __init__(self):
        self.vector_store = load_vector_store()
        # Increased k=12 to handle multiple release comparisons effectively
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 12})
        self.llm = self._load_llm()
        self.rag_chain = self._build_chain()

    def _load_llm(self):
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, device_map="auto", torch_dtype="auto")

        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.01, # Minimal temperature for technical accuracy
            do_sample=True,
            return_full_text=False 
        )
        return HuggingFacePipeline(pipeline=text_gen_pipeline)

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_template(
            """You are a technical 3GPP Standards Assistant. 
The context below contains information tagged with specific 3GPP Releases (Rel-15 to Rel-20).

Rules:
1. Answer the question ONLY based on the provided Context.
2. If a specific Release is mentioned in the question, look ONLY for segments tagged with that Release.
3. If no Release is specified, provide the answer based on the LATEST Release (e.g., Release 19) found in the context.
4. When comparing versions, clearly state what was added or changed between the Release tags (e.g., "In Rel-15... but in Rel-17...").
5. If the information is not present in the provided context for a specific release, say: "I don't know based on the provided standards for that release."
6. Do not use your own internal knowledge if it contradicts the provided context tags.

Context:
{context}

Question:
{question}

Answer:"""
        )

        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def ask(self, question: str) -> str:
        return self.rag_chain.invoke(question)

if __name__ == "__main__":
    bot = ChatBot()
    # Test Question
    response = bot.ask("What are the primary differences in the QoS model between Release 15 and Release 18?")
    print(response)