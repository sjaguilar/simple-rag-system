from langchain.schema import Document  # Import LangChain's built-in Document class
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings  # Updated import
from langchain_community.vectorstores import Chroma
import ollama
import streamlit as st
import os
from io import StringIO

# Set USER_AGENT environment variable (optional but recommended)
os.environ["USER_AGENT"] = "SimpleRAGSystem/1.0"

# Step 1: Load documents from the uploaded file
def load_uploaded_file(uploaded_file):
    # Read the uploaded file contents
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    text = stringio.read()
    # Return a list of LangChain Document objects
    return [Document(page_content=text, metadata={"source": uploaded_file.name})]

# Step 2: Split documents into smaller chunks
def split_documents(docs, chunk_size=500, chunk_overlap=10):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

# Step 3: Generate embeddings and create a vector database
def create_vectorstore(documents, model_name="llama3"):
    embeddings = OllamaEmbeddings(model=model_name)  # Updated to use the new library
    return Chroma.from_documents(documents=documents, embedding=embeddings)

# Step 4: RAG retrieval and response generation
def rag_chain(vectorstore, question):
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(question)
    combined_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    response = ollama.chat(model='llama3', messages=[
        {'role': 'user', 'content': f"Question: {question}\n\nContext: {combined_context}"}
    ])
    return response['message']['content']

# Streamlit Interface
def main():
    st.title("RAG System with Local Text File and Ollama")
    
    # File input for local text file
    uploaded_file = st.file_uploader("Upload a text file:", type="txt")
    question = st.text_input("Ask a question:")
    
    if uploaded_file and st.button("Load and Process"):
        try:
            # Load and process the uploaded file
            docs = load_uploaded_file(uploaded_file)
            splits = split_documents(docs)
            vectorstore = create_vectorstore(splits)
            st.success("File loaded and processed!")
            st.session_state.vectorstore = vectorstore  # Save vectorstore to session state
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    if question and "vectorstore" in st.session_state:
        try:
            answer = rag_chain(st.session_state.vectorstore, question)
            st.write(answer)
        except Exception as e:
            st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
