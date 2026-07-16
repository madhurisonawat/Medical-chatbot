from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

load_dotenv() 

app = Flask(__name__)

# Keep this global, but initialized as None so the BUILD process passes successfully
chain = None

def get_rag_chain():
    global chain
    
    # This block will ONLY run once, on the very first chat request
    if chain is None:
        pinecone_key = os.getenv("PINECONE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        
        embeddings = download_embeddings()

        # Initialize Pinecone safely at RUNTIME
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index("medical-chatbot")

        docsearch = PineconeVectorStore(
            index=index,
            embedding=embeddings
        )

        retriever = docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        llm = ChatGroq(
            model="qwen/qwen3-32b",
            reasoning_effort="none",
            groq_api_key=groq_key
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        qa = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, qa)
        
    return chain


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Message:", msg)
    
    # Safely fetches the active runtime chain
    initialized_chain = get_rag_chain()
    
    response = initialized_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)