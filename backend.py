import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

app = FastAPI(title="PDF RAG API USING Gemma 3 1B")

LM_STUDIO_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"

# Global state holding the active retriever
rag_state = {"retriever": None, "filename": None}

class ChatRequest(BaseModel):
    query: str

def get_embeddings():
    return OpenAIEmbeddings(
        base_url=LM_STUDIO_URL,
        api_key=API_KEY,
        model="nomic-embed-text-v1.5",
        check_embedding_ctx_length=False
    )

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        splits = text_splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=get_embeddings()
        )
        rag_state["retriever"] = vectorstore.as_retriever(search_kwargs={"k": 3})
        rag_state["filename"] = file.filename

        return {"message": f"Successfully indexed '{file.filename}' ({len(splits)} chunks)."}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/chat")
async def chat(request: ChatRequest):
    if rag_state["retriever"] is None:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")

    relevant_docs = rag_state["retriever"].invoke(request.query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    llm = ChatOpenAI(
        base_url=LM_STUDIO_URL,
        api_key=API_KEY,
        model="gemma-3-1b-it",
        temperature=0.2
    )

    prompt = f"""
    		Use only the provided context below to answer the question.
    		If the answer is not in the context, say "I do not know based on the provided PDF."

		Context:{context}
		Question: {request.query}

		Answer:
	   """

    response = llm.invoke(prompt)
    sources = list({doc.metadata.get("page", 0) + 1 for doc in relevant_docs})

    return {
        "answer": response.content,
        "sources": sorted(sources)
    }
