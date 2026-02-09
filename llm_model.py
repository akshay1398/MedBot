from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKENS=os.getenv('HF_TOKEN')
YOUR_GROQ_API_KEY=os.getenv('GROQ_API_KEY')


from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

llm = ChatGroq(
groq_api_key=YOUR_GROQ_API_KEY,
model_name="llama-3.1-8b-instant"
)
    
DB_FAISS_PATH=r'D:\ml\rag\vectorstore'





custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
if you dont know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context
context:{context}
question:{question}

start the anser directly. No small talk please.


"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=['context','question'])

    return prompt


from langchain_community.vectorstores import FAISS

DB_FAISS_PATH=r'D:\ml\rag\vectorstore'
embedding_model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
db=FAISS.load_local(DB_FAISS_PATH,embedding_model, allow_dangerous_deserialization=True
)



# Create QA chain

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(custom_prompt_template)}
)


# Invoke with a single query 
user_query = input("write Query here: ")
response = qa_chain.invoke({"query": user_query})
print("BOT:", response["result"])
