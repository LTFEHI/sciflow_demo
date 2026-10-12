from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIR = "./vector_db"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    # 优化后的中文分块参数
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=140,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    docs = text_splitter.split_documents(documents)
    return docs

def init_vector_db(docs):
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    return db

def get_retriever():
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever
