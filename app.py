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

# 1. Load variables immediately so they are ready for global initialization
load_dotenv() 

# 2. Bind the keys to os.environ right away so LangChain can see them at startup
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

app = Flask(__name__)

# 3. Define the chain initialization function
def initialize_rag_chain():
    embeddings = download_embeddings()

    docsearch = PineconeVectorStore.from_existing_index(
        index_name="medical-chatbot",
        embedding=embeddings
    )

    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    llm = ChatGroq(
        model="qwen/qwen3-32b",
        reasoning_effort="none",
        groq_api_key=os.environ["GROQ_API_KEY"]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    qa = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa)

# 4. Initialize globally! The model will download when Railway boots up the app
chain = initialize_rag_chain()


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Message:", msg)
    
    # 5. The chain is already loaded globally, so the response will be blazing fast
    response = chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    # Dynamic port allocation for production environments
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)