# RAG_ChatBot.py

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
import os

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"  # fallback for local dev
)

class ChatBot:
    def __init__(self):
        # -----------------------------
        # 1. Load documents
        # -----------------------------
        loader = TextLoader("./materials/torontoTravelAssistant.txt", encoding="utf-8")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=150
        )
        docs = splitter.split_documents(documents)

        # -----------------------------
        # 2. Embeddings (Ollama)
        # -----------------------------
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=OLLAMA_BASE_URL
        )

        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(docs)

        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        # -----------------------------
        # 3. LLM (Ollama)
        # -----------------------------
        llm = ChatOllama(
            model="llama3",
            temperature=0.2,
            base_url=OLLAMA_BASE_URL
        )

        # -----------------------------
        # 4. Prompt
        # -----------------------------
        template = """
        You are a Toronto travel assistant.
        Answer the question using ONLY the provided context.
        If the answer is not in the context, say "I don't know".

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

        # -----------------------------
        # 5. RAG Chain
        # -----------------------------
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False
        )


# Optional local test
if __name__ == "__main__":
    bot = ChatBot()
    print(bot.rag_chain.invoke("What are the best places to visit in Toronto?"))
