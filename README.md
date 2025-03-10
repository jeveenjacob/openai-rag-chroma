# openai-rag-chroma

# OpenAI RAG with ChromaDB

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **OpenAI GPT** and **ChromaDB** for storing and retrieving text from documents (PDFs and DOCX). The system extracts content, embeds it, and retrieves relevant information to generate AI-powered responses.

---
```
pip install -r requirements.txt

mkdir documents
mv sample.pdf sample.docx documents/

python extract_and_store.py

python query_chroma.py

python generate_response.py

python api.py

curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d '{"query": "Explain how to instrument with opentelemetry in python application "}'
