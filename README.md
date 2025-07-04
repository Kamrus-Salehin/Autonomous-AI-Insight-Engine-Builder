This project is a FastAPI-based Question Answering (QA) system that uses LangChain's `RetrievalQA` with Cohere (via `langchain_cohere`) and a FAISS vectorstore for semantic document retrieval.

## Features
- Loads `.txt` documents from `./data`
- Creates or loads a local FAISS vectorstore with HuggingFace embeddings
- Serves a FastAPI endpoint to query document knowledge
- Uses Cohere Chat API for accurate and concise answers

## Installation
1. Clone the repository to the local machine and navigate to the project folder:
```bash
git clone https://github.com/Kamrus-Salehin/Autonomous-AI-Insight-Engine-Builder.git
cd Autonomous-AI-Insight-Engine-Builder
```
2. Set up a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
## Environment Setup
Before running the app, create a `.env` file in the root directory:
```bash
COHERE_API_KEY='your_COHERE_API_KEY_here'
```

## Usage
1. Add `.txt` files to the `./data` directory.
2. Run the API server:
```bash
uvicorn main:app --reload
```
3. Access Swagger UI at:
```bash
http://localhost:8000/docs  # 8000 is the port number
```
4. Scroll to the `POST /query` section and expand it by clicking on it.
5. Click the `Try it out` button on the right.
6. In the `request body` area, replace the example text with something like:
```json
{
  "query": "What's the length of the Nile River?"
}
```
7. Hit the `Execute` button.
8. The response will appear right below, including:
- The answer generated by Gemini from provided documents
- The list of sources (file names) used to generate the answer

## Notes
- The first run will embed and index documents into FAISS under `vectorstore_index/`.
- Queries are answered with context from the uploaded `.txt` files only.
