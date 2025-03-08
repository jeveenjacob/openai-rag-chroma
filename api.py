from flask import Flask, request, jsonify
from generate_response import generate_response

app = Flask(__name__)

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    query_text = data.get("query", "")
    if not query_text:
        return jsonify({"error": "Query text is required"}), 400
    
    response = generate_response(query_text)
    return jsonify({"query": query_text, "response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
