from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import fitz  # PyMuPDF
import docx
import io
import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from openai import OpenAI


# ---------------------------
# FASTAPI APP
# ---------------------------

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:5173/",
    "http://localhost:8080",
    "http://localhost:8080/",
    "https://resumexai.netlify.app",
    "https://resumexai.netlify.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# OPENAI CLIENT
# ---------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------
# ANALYSIS RESULT MODEL
# ---------------------------

class AnalysisResult(BaseModel):
    score: int
    skills: list[str]
    summary: str
    weaknesses: list[str]
    suggestions: list[str]
    improvedResume: str
    strengths: list[str] = []   # Added for completeness
    sections: list = []         # Optional


# ---------------------------
# PDF & DOCX TEXT EXTRACTION
# ---------------------------

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


# ---------------------------
# AI ANALYSIS FUNCTION
# ---------------------------

def analyze_with_ai(resume_text: str) -> AnalysisResult:
    prompt = f"""
You are a professional resume reviewer.

Analyze this resume and return STRICT JSON with keys:
- score (0-100)
- skills (list of skills)
- summary (text)
- strengths (list)
- weaknesses (list)
- suggestions (list)
- improvedResume (full improved resume)

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

    data = json.loads(response.choices[0].message.content)
    return AnalysisResult(**data)


# ---------------------------
# API: UPLOAD + ANALYZE RESUME
# ---------------------------

@app.post("/analyze-resume", response_model=AnalysisResult)
async def analyze_resume(file: UploadFile = File(...)):
    file_bytes = await file.read()
    fname = file.filename.lower()

    if fname.endswith(".pdf"):
        resume_text = extract_text_from_pdf(file_bytes)

    elif fname.endswith(".docx"):
        resume_text = extract_text_from_docx(file_bytes)

    else:
        return {"error": "Only PDF or DOCX files are supported"}

    if not resume_text.strip():
        return {"error": "Could not read text from file"}

    result = analyze_with_ai(resume_text)
    return result


# ---------------------------
# API: DOWNLOAD PDF REPORT
# ---------------------------

@app.post("/download-report")
async def download_report(data: AnalysisResult):
    """
    Generates a PDF from the resume analysis.
    """

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []

    # Title
    elements.append(Paragraph("<b>Resume Analysis Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.2 * inch))

    # Score
    elements.append(Paragraph(f"<b>Score:</b> {data.score}/100", styles["Heading2"]))
    elements.append(Spacer(1, 0.1 * inch))

    # Summary
    elements.append(Paragraph("<b>Summary</b>", styles["Heading3"]))
    elements.append(Paragraph(data.summary, styles["BodyText"]))
    elements.append(Spacer(1, 0.2 * inch))

    # Strengths
    if data.strengths:
        elements.append(Paragraph("<b>Strengths</b>", styles["Heading3"]))
        for s in data.strengths:
            elements.append(Paragraph(f"• {s}", styles["BodyText"]))
        elements.append(Spacer(1, 0.2 * inch))

    # Weaknesses
    elements.append(Paragraph("<b>Weaknesses</b>", styles["Heading3"]))
    for w in data.weaknesses:
        elements.append(Paragraph(f"• {w}", styles["BodyText"]))
    elements.append(Spacer(1, 0.2 * inch))

    # Skills
    elements.append(Paragraph("<b>Skills</b>", styles["Heading3"]))
    for skill in data.skills:
        elements.append(Paragraph(f"• {skill}", styles["BodyText"]))
    elements.append(Spacer(1, 0.2 * inch))

    # Suggestions
    elements.append(Paragraph("<b>Suggestions</b>", styles["Heading3"]))
    for sug in data.suggestions:
        elements.append(Paragraph(f"• {sug}", styles["BodyText"]))
    elements.append(Spacer(1, 0.3 * inch))

    # Improved Resume
    elements.append(Paragraph("<b>Improved Resume</b>", styles["Heading3"]))
    elements.append(Paragraph(data.improvedResume.replace("\n", "<br/>"), styles["BodyText"]))

    doc.build(elements)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=resume_report.pdf"},
    )


# ---------------------------
# ROOT ROUTE
# ---------------------------

@app.get("/")
def home():
    return {"message": "Resume AI Backend Running Successfully!"}
