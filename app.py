from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pathlib import Path
import os
import uvicorn

# ----------------------
# Initialize FastAPI app
# ----------------------
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# ----------------------
# Load environment variables
# ----------------------
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ----------------------
# Initialize embeddings and retriever
# ----------------------
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ----------------------
# Initialize LLM
# ----------------------
chat_model = ChatOpenAI(model="gpt-4o")

# ----------------------
# Setup RAG chain (no memory)
# ----------------------
# Combine docs chain for final answer
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
combine_docs_chain = create_stuff_documents_chain(llm=chat_model, prompt=qa_prompt)

# Retrieval chain
qa_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combine_docs_chain
)

# ----------------------
# FastAPI routes
# ----------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/get")
async def chat(msg: str = Form(...)):
    print("User Input:", msg)

    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(msg)
    unique_sources = {doc.metadata.get("source", "Unknown source") for doc in docs}
    sources = list(unique_sources)

    # Invoke RAG chain
    response = qa_chain.invoke({"input": msg})
    answer = response["answer"]

    print("Response:", answer)
    print("Sources:", sources)

    return {"answer": answer, "sources": sources}

# ----------------------
# Run server
# ----------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)
