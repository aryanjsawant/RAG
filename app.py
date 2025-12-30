# app.py
from RAG_ChatBot import ChatBot
import streamlit as st
import re

bot = ChatBot()

st.title('Electrical Department Assistant Bot')

# Function for generating LLM response
def generate_response(user_input):
    raw_response = bot.rag_chain.invoke(user_input)

    # Extract only text after "Answer:"
    match = re.search(r"answer:\s*(.*)", raw_response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback if "Answer:" is missing
    return raw_response.strip()


# Initialize chat messages and typing state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! What do you want to learn today?"}
    ]

if "is_typing" not in st.session_state:
    st.session_state.is_typing = False


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Disable input while bot is typing
if st.session_state.is_typing:
    st.chat_input("Bot is typing...", disabled=True)
else:
    if user_input := st.chat_input():
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.write(user_input)

        st.session_state.is_typing = True

        with st.chat_message("assistant"):
            with st.spinner("Getting your answer from uploaded documents..."):
                response = generate_response(user_input)
                st.write(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        st.session_state.is_typing = False
