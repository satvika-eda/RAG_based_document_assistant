from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import shutil
import os
import config

def document_loader():
    return DirectoryLoader(config.DATA_DIR, glob="*.pdf").load()

def doc_splitter():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  
        chunk_overlap=200,  
        length_function=len,  
    )
    return text_splitter.split_documents(docs)

def create_chroma_db():
    if os.path.exists(config.CHROMA_DIR):
        shutil.rmtree(config.CHROMA_DIR)
    embedding_model = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks, embedding_model, persist_directory=config.CHROMA_DIR)

if __name__ == "__main__":
    docs = document_loader()
    chunks = doc_splitter()
    create_chroma_db()
