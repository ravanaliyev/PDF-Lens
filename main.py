import shutil
import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Any
from parser import PDFParser, PDFParserError

app = FastAPI(title="PDF Analysis API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development (e.g., Flutter)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Create a temporary file to save the upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        try:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    try:
        # Initialize parser
        parser = PDFParser(tmp_path, config_path="config.json")
        
        # Analyze
        metadata = parser.get_metadata()
        topics = parser.get_text_by_topics()
        
        # Format response
        response_data = {
            "filename": file.filename,
            "metadata": metadata,
            "topic_count": len(topics),
            "topics": [
                {
                    "title": t.get("topic_title"),
                    "level": t.get("level", 1),
                    "page_start": t.get("page_number", 0), # Note: topic segmentation might not have exact page in output dict yet, need to check parser.py
                    "content_preview": t.get("content", "")[:200]
                }
                for t in topics
            ]
        }
        
        # Add full content if needed or just basics. 
        # User requested: metadata, topics with levels.
        # parser.get_text_by_topics() returns: topic_title, start_line, end_line, content, level.
        # It doesn't strictly return page number for each topic unless we map it back, 
        # but topics are sequential.
        
        return JSONResponse(content=response_data)

    except PDFParserError as e:
        raise HTTPException(status_code=400, detail=f"PDF Parsing Error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        parser.close()

@app.get("/")
def read_root():
    return {"message": "PDF Analysis API is running"}
