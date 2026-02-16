from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os 
load_dotenv()

GROQ_KEY = os.getenv("GROQ_KEY")
VECTORSTORE_PATH = r'Vectorstore'

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(VECTORSTORE_PATH,embedding_model,allow_dangerous_deserialization=True)


custome_prompt = """
You are a helpful AI assistant.

Use ONLY the information provided in the context below to answer the user's question.

Rules:
- Do NOT use your own knowledge.
- Do NOT make up information.
- If the answer is not present in the context, say:
  "I could not find the answer in the provided document."
- Keep the answer clear and concise.
- If possible, quote relevant parts from the context.

Context:
{context}

Question:
{question}

Answer:
"""

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


prompt = PromptTemplate(template=custome_prompt, input_variables=['context','question'])

llm = ChatGroq(groq_api_key = GROQ_KEY ,model="LLaMA-3.1-8B-Instant",temperature=0)

qachain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type='stuff',
    chain_type_kwargs={"prompt":prompt}
)


query = input('Ask Question')

result = qachain.invoke({'query':query})
print(result['result'])
