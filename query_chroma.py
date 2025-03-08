
import chromadb
from langchain_openai import OpenAIEmbeddings
import os

# Load OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="document_collection")

# Function to retrieve documents based on query
def retrieve_docs(query):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
    query_vector = embeddings.embed_query(query)

    # Perform ChromaDB search
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=2,
        include=['metadatas']  # ✅ Ensure we retrieve metadata (text content)
    )

    # Debugging Output - Print raw ChromaDB results
    print("\n🔍 ChromaDB Raw Results:", results, "\n")

    # Extract text content from metadata
    if results and "metadatas" in results and results["metadatas"]:
        docs = []
        for metadata_list in results["metadatas"]:  # ✅ Use metadatas instead of documents
            for metadata in metadata_list:
                if metadata and "text" in metadata:  # ✅ Ensure metadata contains text
                    docs.append({"text": metadata["text"]})
        
        # Debugging Output - Print retrieved documents
        print("✅ Retrieved Documents:", docs, "\n")
        return docs if docs else []

    print("⚠️ No relevant documents found.")
    return []  # Return empty list if no results

if __name__ == "__main__":
    query = input("Enter your query: ")
    retrieved_docs = retrieve_docs(query)
    print("\n Final Retrieved Context:", retrieved_docs)