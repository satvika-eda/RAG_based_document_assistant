import streamlit as st
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None, timeout=None, max_retries=2)

st.title("RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 

if query_text := st.chat_input("Ask your question here ... "):
    with st.chat_message("user"):
        st.markdown(query_text)
    st.session_state.messages.append({"role": "user", "content": query_text})

prediction = None

if query_text:
    prediction = model.predict(query_text)

if prediction is not None:
    response = f"Bot: {prediction}"
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
