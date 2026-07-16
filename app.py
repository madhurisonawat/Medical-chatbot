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

# Remove or comment out load_dotenv() to prevent it from messing with Railway's settings
# load_dotenv() 

app = Flask(__name__)

chain = None

def get_rag_chain():
    global chain
    
    if chain is None:
        # DIAGNOSTIC PRINTS: These will show up in your Railway Console logs
        print("--- DEBUGGING ENVIRONMENT VARIABLES ---")
        print("All available keys in environment:", list(os.environ.keys()))
        print("PINECONE_API_KEY value exists?:", os.getenv("PINECONE_API_KEY") is not None)
        print("---------------------------------------")

        pinecone_key = os.getenv("PINECONE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        
        if not pinecone_key:
            raise ValueError("CRITICAL ERROR: PINECONE_API_KEY is completely empty or missing from the environment!")

        embeddings = download_embeddings()

        # Initialize Pinecone safely
        pc = Pinecone(api_key=pinecone_key.strip()) # .strip() removes accidental spaces
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
            groq_api_key=groq_key.strip() if groq_key else None
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
    if request.is_json:
        data = request.get_json()
        msg = data.get("msg", "")
    else:
        msg = request.form.get("msg", "")
        
    if not msg:
        return "Error: Empty message received", 400
        
    print("User Message:", msg)
    
    try:
        initialized_chain = get_rag_chain()
        response = initialized_chain.invoke({"input": msg})
        return str(response.get("answer", "I couldn't process an answer."))
    except Exception as e:
        print(f"Exception caught in /get route: {e}")
        return f"Internal Server Error: {str(e)}", 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)