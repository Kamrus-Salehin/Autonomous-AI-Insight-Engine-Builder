from fastapi import FastAPI
from pydantic import BaseModel
from llm_agent import get_qa_chain
from retriever import vectorstore

app = FastAPI()
qa_chain = get_qa_chain(vectorstore)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "API is live. Use POST /query"}

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    result = qa_chain.invoke(request.query)
    answer = result["result"]
    sources = [doc.metadata.get("source", "") for doc in result["source_documents"]]
    return {"answer": answer, "sources": sources}