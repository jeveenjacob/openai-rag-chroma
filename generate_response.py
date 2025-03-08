import openai
import os
from query_chroma import retrieve_docs

# Load OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def generate_response(query):
    context = retrieve_docs(query)

    # If no relevant documents found, provide an alternative response
    if not context:
        context_text = "I couldn't find relevant information in the document. However, based on general knowledge:"
    else:
        context_text = " ".join([doc["text"] for doc in context])

    # Improved Prompt Engineering: Structure AI responses
    prompt = f"""
    You are an expert in OpenTelemetry, observability, and Python instrumentation. 
    Answer the following query using the format below:

    ### Summary:
    (Provide a short summary in 2-3 sentences)

    ### Detailed Explanation:
    (Provide an in-depth response based strictly on the document content below)

    ### Example:
    (If applicable, include a relevant code snippet or real-world example)

    --- DOCUMENT CONTENT ---
    {context_text}
    ------------------------

    **Important Instructions:**
    - Do NOT make up information. If the document does not contain an answer, say: "The document does not provide this information."
    - Provide step-by-step instructions when needed.
    - Format code snippets properly if included.
    - make sure to add mandatory example attributes as well in your suggestions.
    
    **Query:** {query}
    """

    response = client.chat.completions.create(  # ✅ Uses OpenAI's latest API format
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are an AI assistant."},
                  {"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content  # ✅ Extracts text correctly

if __name__ == "__main__":
    query_text = input("Ask your question: ")
    print(generate_response(query_text))
