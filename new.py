import os
import streamlit as st
# from langchain.nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
# from langchain.output_parsers import StrOutputParser
import pickle

# Set page configuration
st.set_page_config(layout="wide")

# Directory for uploaded documents
DOCS_DIR = os.path.abspath("./uploaded_docs")
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)
 
# Sidebar for uploading documents
with st.sidebar:
    st.subheader("Add to the Knowledge Base")
    with st.form("my-form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload a file to the Knowledge Base:", accept_multiple_files=True)
        submitted = st.form_submit_button("Upload!")
        if submitted and uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(DOCS_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                st.success(f"File {uploaded_file.name} uploaded successfully!")

# AI model setup
nvidia_api_key = "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"
llm = ChatNVIDIA(model="mixtral_8x7b", api_key=nvidia_api_key)

# llm = ChatNVIDIA(model="mixtral_8x7b")
document_embedder = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="passage",api_key=nvidia_api_key)
query_embedder = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="query",api_key=nvidia_api_key)

# Check for existing vector store
vector_store_path = "vectorstore.pkl"
vector_store_exists = os.path.exists(vector_store_path)
use_existing_vector_store = st.radio("Use existing vector store", ["Yes", "No"])

# Load documents and process them
raw_documents = DirectoryLoader(DOCS_DIR).load()
vector_store = None
if use_existing_vector_store == "Yes" and vector_store_exists:
    with open(vector_store_path, "rb") as f:
        vector_store = pickle.load(f)
else:
    if raw_documents:
        text_splitter = CharacterTextSplitter(chunk_size=2000, overlap=200)
        documents = text_splitter.split_documents(raw_documents)
        vector_store = FAISS.from_documents(documents, document_embedder)
        with open(vector_store_path, "wb") as f:
            pickle.dump(vector_store, f)

# Chat interface
st.subheader("Chat with your AI Assistant, Envie!")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask your question:")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    if vector_store:
        retriever = vector_store.as_retriever()
        context = "\n\n".join(doc.page_content for doc in retriever.get_relevant_documents(user_input))
        augmented_user_input = f"Context: {context}\n\nQuestion: {user_input}"
        response = llm.run({"input": augmented_user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
