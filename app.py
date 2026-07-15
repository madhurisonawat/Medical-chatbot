from flask import Flask
from langchain_groq import ChatGroq
import os
from src.helper import download_embeddings

app = Flask(__name__)

llm = ChatGroq(
    model="qwen/qwen3-32b",
    reasoning_effort="none"
)


embeddings = download_embeddings()
@app.route("/")
def home():
    return "Groq Loaded Successfully!"

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)