import os
import chromadb
from langchain.document_loaders import PyPDFLoader
from docx import Document
from langchain_openai import OpenAIEmbeddings

# Load OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="document_collection")

# Function to extract text from a DOCX file
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

# Function to split text into chunks
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to process and store documents
def store_documents(folder_path="/Users/jeveenjacob/Documents/GitHub/openai-rag-chroma/documents"):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            text = " ".join([doc.page_content for doc in documents])
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        else:
            continue
        
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            embedding_vector = embeddings.embed_query(chunk)
            collection.add(
                ids=[f"{filename}-{i}"],
                embeddings=[embedding_vector],
                metadatas=[{"filename": filename, "text": chunk}]
            )
            print(f"Stored chunk {i} from {filename}")

    print(f"Total documents stored: {collection.count()}")

if __name__ == "__main__":
    store_documents()