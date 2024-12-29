from dotenv import  load_dotenv
from langchain_community.document_loaders import DirectoryLoader,PyPDFDirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from embedding_function import GeminiEmbeddingFunction
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import os
import shutil


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_PATH = os.getenv("CHROMA_PATH")
DATA_PATH = "data"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    # loader = DirectoryLoader(DATA_PATH, glob = "*.md")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    docs = loader.load()
    return docs

def split_text(docs:list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 400,
        length_function = len,
        add_start_index = True
    )
    chunks = splitter.split_documents(docs)
    print(f"Split {len(docs)} into {len(chunks)} chunks")
    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    db = Chroma.from_documents(
        chunks, GoogleGenerativeAIEmbeddings(model= "models/text-embedding-004"), persist_directory = CHROMA_PATH
    )
    # db = Chroma(persist_directory = CHROMA_PATH, embedding_function = GeminiEmbeddingFunction())
    # db.add(chunks)
    db.persist()
    print(f"Saved {len(chunks)} chunks in db at {CHROMA_PATH}")


if __name__ == "__main__" :
    main()