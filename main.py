from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import fitz  # PyMuPDF
import docx
import io
import os
import json
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import requests
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# ---------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:8080",
    "https://resumexai.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------
# Pydantic Model
# ---------------------------------------------------

class AnalysisResult(BaseModel):
    score: int
    skills: list[str]
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]
    improvedResume: str


# ---------------------------------------------------
# File Extraction
# ---------------------------------------------------

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


# ---------------------------------------------------
# DeepSeek AI ANALYSIS
# ---------------------------------------------------

def analyze_with_deepseek(text: str) -> AnalysisResult:
    url = "https://api.deepseek.com/chat/completions"

    prompt = f"""
You are a professional resume reviewer.
Analyze the resume below and return a STRICT JSON object with EXACT keys:

score: number 0-100
skills: list of strings
summary: string
strengths: list of strings
weaknesses: list of strings
suggestions: list of strings
improvedResume: string

Resume Text:
\"\"\"{text}\"\"\"
"""

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "deepseek-chat",
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are an expert ATS resume analyzer."},
            {"role": "user", "content": prompt},
        ]
    }

    response = requests.post(url, headers=headers, json=payload, timeout=200)
    data = response.json()

    if "choices" not in data:
        raise Exception("DeepSeek Error: " + str(data))

    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)

    return AnalysisResult(**parsed)


# ---------------------------------------------------
# API: Upload & Analyze
# ---------------------------------------------------

@app.post("/analyze-resume", response_model=AnalysisResult)
async def analyze_resume(file: UploadFile = File(...)):
    file_bytes = await file.read()
    name = file.filename.lower()

    if name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(file_bytes)
    elif name.endswith(".docx"):
        resume_text = extract_text_from_docx(file_bytes)
    else:
        return {"error": "Only PDF or DOCX accepted"}

    if not resume_text.strip():
        return {"error": "Could not extract text"}

    return analyze_with_deepseek(resume_text)


# ---------------------------------------------------
# API: ULTRA PREMIUM PDF REPORT
# ---------------------------------------------------

@app.post("/download-report")
async def download_report(data: AnalysisResult):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=45,
        rightMargin=45,
        topMargin=60,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "title",
        parent=styles["Title"],
        fontSize=26,
        leading=32,
        alignment=1,
        textColor="#C9A227",
    )

    header_style = ParagraphStyle(
        "header",
        parent=styles["Heading2"],
        fontSize=17,
        leading=22,
        textColor="#2E2E2E",
    )

    body_style = ParagraphStyle(
        "body",
        parent=styles["BodyText"],
        fontSize=11.5,
        leading=16,
        textColor="#333",
    )

    bullet_style = ParagraphStyle(
        "bullet",
        parent=styles["BodyText"],
        fontSize=11.5,
        leftIndent=15,
        leading=16,
    )

    elements = []

    # Title
    elements.append(Paragraph("Resume Analysis Report", title_style))
    elements.append(Spacer(1, 0.4 * inch))

    # Score Box
    table = Table(
        [[Paragraph(f"<b>Resume Score: {data.score}/100</b>", header_style)]],
        colWidths=[6.2 * inch]
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.Color(0.95, 0.88, 0.55)),
        ("BOX", (0, 0), (-1, -1), 2, colors.Color(0.8, 0.65, 0.15)),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))

    # Summary
    elements.append(Paragraph("Professional Summary", header_style))
    elements.append(Paragraph(data.summary.replace("\n", "<br/>"), body_style))
    elements.append(Spacer(1, 0.3 * inch))

    # Strengths
    elements.append(Paragraph("Key Strengths", header_style))
    for s in data.strengths:
        elements.append(Paragraph(f"• {s}", bullet_style))
    elements.append(Spacer(1, 0.3 * inch))

    # Weaknesses
    elements.append(Paragraph("Areas to Improve", header_style))
    for w in data.weaknesses:
        elements.append(Paragraph(f"• {w}", bullet_style))
    elements.append(Spacer(1, 0.3 * inch))

    # Skills
    elements.append(Paragraph("Detected Skills", header_style))
    for skill in data.skills:
        elements.append(Paragraph(f"• {skill}", bullet_style))
    elements.append(Spacer(1, 0.3 * inch))

    # Suggestions
    elements.append(Paragraph("Suggestions", header_style))
    for sug in data.suggestions:
        elements.append(Paragraph(f"• {sug}", bullet_style))

    elements.append(PageBreak())

    # Improved Resume Page
    elements.append(Paragraph("AI-Optimized Resume", title_style))
    elements.append(Paragraph(data.improvedResume.replace("\n", "<br/>"), body_style))

    doc.build(elements)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=ResumeAI_Premium_Report.pdf"},
    )


# ---------------------------------------------------
# Root
# ---------------------------------------------------

@app.get("/")
def root():
    return {"message": "DeepSeek Resume AI Backend Running!"}
