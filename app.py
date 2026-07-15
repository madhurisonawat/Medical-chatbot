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

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)


rag_chain = None

def get_rag_chain():
    global rag_chain

    if rag_chain is None:
        embeddings = download_embeddings()

        docsearch = PineconeVectorStore.from_existing_index(
            index_name="medical-chatbot",
            embedding=embeddings
        )

        retriever = docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={"k":3}
        )

        llm = ChatGroq(
            model="qwen/qwen3-32b",
            reasoning_effort="none"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human","{input}")
        ])

        qa = create_stuff_documents_chain(llm,prompt)

        rag_chain = create_retrieval_chain(retriever,qa)

    return rag_chain


@app.route("/")
def index():
    return render_template('chat.html')

chain = get_rag_chain()


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)