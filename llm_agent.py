import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

load_dotenv()

prompt_template = PromptTemplate.from_template("""
You are an AI assistant. Use **only** the following context to answer the question. 
Do not make up facts or use any external knowledge. Try to generate summarized answer using bullet points.

Context:
{context}

Question:
{question}

Answer:
""")

def get_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ["api_key"],
        temperature=0.1
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
