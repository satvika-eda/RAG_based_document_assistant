
import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import config
from langchain_core.prompts import ChatPromptTemplate

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

def query():
    embedding_model = OpenAIEmbeddings()
    db = Chroma(persist_directory=config.CHROMA_DIR, embedding_function=embedding_model)
    result = db.similarity_search_with_relevance_scores(query_text, k=5)
    if len(result) == 0:
        print("I don't know")
        return
    else:
        return "\n ------ \n".join([doc.page_content for doc, score in result])

if query_text:
    context = query()
    prompt_template = ChatPromptTemplate.from_template(config.PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, query=query_text)
    prediction = model.predict(prompt)

if prediction is not None:
    response = f"Bot: {prediction}"
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
