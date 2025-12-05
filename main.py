from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz  # PyMuPDF
import docx
import io
import os
from openai import OpenAI

app = FastAPI()

origins = [
    "http://localhost:5173",
    "https://resumexai.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# const API_URL = "https://resume-backend-hkeq.onrender.com/"
# API_KEY = "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"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class AnalysisResult(BaseModel):
    score: int
    skills: list[str]
    summary: str
    weaknesses: list[str]
    suggestions: list[str]
    improvedResume: str

def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(file_bytes: bytes) -> str:
    file_stream = io.BytesIO(file_bytes)
    document = docx.Document(file_stream)
    return "\n".join(p.text for p in document.paragraphs)

def analyze_with_ai(resume_text: str) -> AnalysisResult:
    prompt = f"""
You are a professional resume reviewer.

Analyze this resume and return JSON with:
- score (0-100)
- skills (list)
- summary
- weaknesses (list)
- suggestions (list)
- improvedResume

Resume:
\"\"\"{resume_text}\"\"\"
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an expert resume analyzer."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"}
    )

    # Extract JSON safely
    import json
    data = json.loads(response.choices[0].message.content)

    return AnalysisResult(**data)


@app.post("/analyze-resume", response_model=AnalysisResult)
async def analyze_resume(file: UploadFile = File(...)):
    file_bytes = await file.read()
    fname = file.filename.lower()

    if fname.endswith(".pdf"):
        resume_text = extract_text_from_pdf(file_bytes)
    elif fname.endswith(".docx"):
        resume_text = extract_text_from_docx(file_bytes)
    else:
        return {"error": "Only PDF or DOCX supported"}

    if not resume_text.strip():
        return {"error": "Could not read text"}

    result = analyze_with_ai(resume_text)
    return result
