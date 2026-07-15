from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs:List[Document]) -> List[Document]:
    minimal_docs: List[Document]=[]
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_doc=Document(page_content=doc.page_content, metadata={"source":src})
        minimal_docs.append(minimal_doc)
    return minimal_docs
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts = text_splitter.split_documents(minimal_docs)
    return texts


def download_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings