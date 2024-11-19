# Simple RAG System

A simple Retrieval-Augmented Generation (RAG) system built with Streamlit, Ollama, and LangChain.

## Features
- Upload local `.txt` files
- Ask questions about the document's content
- Powered by `llama3` and LangChain embeddings

## Prerequisites
- Python 3.8+
- Ollama installed and configured
- `llama3` model pulled via Ollama (`ollama pull llama3`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/simple-rag-system.git
   cd simple-rag-system
2. Install dependencies:
   ```bash
   pip install -r requirements.txt   
4. Run the application:
   ```bash
   streamlit run rag_system.py

## Usage
1. Open the Streamlit interface in your browser.
2. Upload a .txt file containing text.
3. Ask questions based on the file's content.




