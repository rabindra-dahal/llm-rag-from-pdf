import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from rag_engine import PDFRagEngine

app = FastAPI(title="PDF RAG API USING Gemma 3 1B")

# Initialize the core engine instance
rag_engine = PDFRagEngine()


class ChatRequest(BaseModel):
    query: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunk_count = rag_engine.index_pdf(
            file_path=tmp_path,
            original_filename=file.filename,
        )
        return {
            "message": f"Successfully indexed '{file.filename}' ({chunk_count} chunks)."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        return rag_engine.ask(request.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
