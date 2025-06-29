import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_cohere import ChatCohere
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
    llm = ChatCohere(
        COHERE_API_KEY=os.environ["COHERE_API_KEY"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
