
from langchain_community.document_loaders import DirectoryLoader   , PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os 


# API AND PATHS 
HF_EMBEDDING = os.getenv("HF_EMBEDDING")
GROQ_KEY = os.getenv("GROQ_KEY")
DATA_PATH = r"Data"
VECTORSTORE_PATH = r'Vectorstore'

# LOAD DOCUMNETS 

loader = DirectoryLoader(DATA_PATH,glob='*.pdf', loader_cls=PyPDFLoader)
document = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)

text_chunks = text_splitter.split_documents(document)

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.from_documents(texts,_chunkembedding=embedding_model)
db.save_local(VECTORSTORE_PATH)


