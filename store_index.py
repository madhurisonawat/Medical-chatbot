from dotenv import load_dotenv
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from src.helper import (load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings)

load_dotenv()

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

extracted_data = load_pdf_files(data="data/")
filtered_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filtered_data)
embeddings = download_embeddings()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name="medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index=pc.index(index_name)
docsearch = PineconeVectorStore.from_documents(documents=text_chunks, embedding=embeddings, index_name=index_name)