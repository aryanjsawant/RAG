# app.py

import streamlit as st
from RAG_ChatBot import ChatBot
import shutil
import os

st.set_page_config(page_title="Electrical Dept Assistant")

st.title("⚡ Electrical Department Assistant")

# Sidebar controls
with st.sidebar:
    if st.button("🔄 Reindex New Documents"):
        st.cache_resource.clear()
        st.success("Reindexed successfully")

    if st.button("🗑️ Reset Vector Database"):
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        st.cache_resource.clear()
        st.warning("Vector database reset")

# Initialize bot
bot = ChatBot()

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me about electrical standards."}
    ]

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Searching standards..."):
            response = bot.chain.invoke(user_input)

            answer = response["result"]
            sources = response["source_documents"]

            citations = set(
                f"{d.metadata['source']}, page {d.metadata['page']}"
                for d in sources
            )

            final_answer = answer
            if citations:
                final_answer += "\n\nSources:\n" + "\n".join(citations)

            st.write(final_answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": final_answer}
    )
