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
You are an expert resume reviewer.

Analyze this resume and return a JSON with:
- score (0-100)
- skills: array
- summary: 2–3 line summary
- weaknesses: list
- suggestions: list
- improvedResume: rewritten resume

Resume text:
\"\"\"{resume_text}\"\"\"
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "resume_analysis",
                "schema": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer"},
                        "skills": {"type": "array", "items": {"type": "string"}},
                        "summary": {"type": "string"},
                        "weaknesses": {"type": "array", "items": {"type": "string"}},
                        "suggestions": {"type": "array", "items": {"type": "string"}},
                        "improvedResume": {"type": "string"}
                    },
                    "required": ["score", "skills", "summary", "weaknesses", "suggestions", "improvedResume"]
                }
            }
        }
    )

    data = response.output[0].parsed
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
