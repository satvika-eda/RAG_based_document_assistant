
import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import config
from langchain_core.prompts import ChatPromptTemplate
import PyPDF2
import shutil
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()

model = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None, timeout=None, max_retries=2)

embeddings = OpenAIEmbeddings()

# Initialize Vector Store (Chroma)
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="vector_store"
)

def doc_splitter(uploaded_file):
    """
    Processes the uploaded file and splits it into Document chunks.

    Args:
        uploaded_file (UploadedFile): The file uploaded via Streamlit's file uploader.

    Returns:
        List[Document]: A list of Document objects containing chunks of text.
    """
    # Initialize text variable
    text = ""

    # Handle PDF files
    if uploaded_file.type == "application/pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
            if not text.strip():
                st.error("**No extractable text found in the PDF.**")
                return []
        except Exception as e:
            st.error(f"**Error extracting text from PDF:** {e}")
            return []

    # Handle Text files
    elif uploaded_file.type.startswith("text"):
        try:
            text = uploaded_file.getvalue().decode("utf-8")
            if not text.strip():
                st.error("**Uploaded text file is empty.**")
                return []
        except Exception as e:
            st.error(f"**Error reading text file:** {e}")
            return []

    # Unsupported file types
    else:
        st.error("**Unsupported file type. Please upload a PDF or text file.**")
        return []

    # Create a Document object
    doc = Document(page_content=text, metadata={"source": uploaded_file.name})

    # Initialize the TextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # Size of each chunk in characters
        chunk_overlap=200,  # Overlap between chunks in characters
        length_function=len,
    )

    # Split the Document into chunks
    chunks = text_splitter.split_documents([doc])

    print(len(chunks))
    return chunks

def add_documents_to_vector_store(chunks):
    """Adds Document chunks to the vector store."""
    if chunks:
        if os.path.exists(config.CHROMA_DIR):
            shutil.rmtree(config.CHROMA_DIR)
        ## TODO : remove old docs
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=config.CHROMA_DIR
        )
        print("added")
        vector_store.add_documents(chunks)

st.title("RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 

attachment_button = st.button("📎 Attach File")

if attachment_button:
    st.session_state.show_uploader = not st.session_state.show_uploader

uploaded_file = None
if st.session_state.show_uploader:
    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["txt", "pdf", "png", "jpg"],
        key="uploader",
        help="Supported formats: .txt, .pdf, .png, .jpg | Max size: 15MB"
    )

if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
    st.session_state.current_file_name = None

def query(q_text):
    embedding_model = OpenAIEmbeddings()
    db = Chroma(persist_directory=config.CHROMA_DIR, embedding_function=embedding_model)
    result = db.similarity_search_with_relevance_scores(q_text, k=5)
    if len(result) == 0:
        print("I don't know")
        return
    else:
        return "\n ------ \n".join([doc.page_content for doc, score in result])

prediction = None

query_text = st.chat_input("Ask your question here ... ")

if query_text or uploaded_file:
    # Handle Text Input
    if query_text:
        # Display User Message
        with st.chat_message("user"):
            st.markdown(query_text)
        # Append to Session State
        st.session_state.messages.append({"role": "user", "content": query_text})
    
    # Handle File Upload
    if uploaded_file:
        # Check if a new file has been uploaded
        if (not st.session_state.file_processed) or (uploaded_file.name != st.session_state.current_file_name):
            # Reset state for new file upload
            st.session_state.file_processed = False
            st.session_state.current_file_name = uploaded_file.name

            # File Size Validation (Max 15MB)
            MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB
            if uploaded_file.size > MAX_FILE_SIZE:
                st.error("🚫 File is too large. Please upload a file smaller than 15MB.")
            else:
                # Process and split the document
                if uploaded_file.type == "application/pdf":
                    chunks = doc_splitter(uploaded_file)
                    if chunks:
                        add_documents_to_vector_store(chunks)  # Add to vector store
                        st.session_state.file_processed = True  # Mark as processed
                        st.success("📂 File processed and added to the vector store!")
                    else:
                        st.warning("⚠️ No valid content to add to the vector store.")
                else:
                    st.error("Unsupported file type. Only PDF files are supported.")

    # Prepare the Prompt for the Model
    prediction = None
    if query_text:
        context = query(query_text)
        prompt_template = ChatPromptTemplate.from_template(config.PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context, query=query_text)
        
        # Get the Model's Prediction with Loading Indicator
        try:
            with st.spinner("🤖 Generating response..."):
                prediction = model.predict(prompt)
        except Exception as e:
            prediction = f"**Error generating response:** {e}"

    # Display Assistant's Response
    if prediction is not None:
        response = f"Bot: {prediction}"
        with st.chat_message("assistant"):
            st.markdown(response)
        # Append to Session State
        st.session_state.messages.append({"role": "assistant", "content": response})
