from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import fitz  # PyMuPDF
import docx
import io
import os
import json
from openai import OpenAI

# ReportLab (PDF)
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch


# ---------------------------------------------------
# FASTAPI SETUP
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
# OPENAI CLIENT
# ---------------------------------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------
# RESPONSE MODEL
# ---------------------------------------------------

class AnalysisResult(BaseModel):
    score: int
    skills: list[str]
    summary: str
    weaknesses: list[str]
    suggestions: list[str]
    improvedResume: str
    strengths: list[str] = []
    sections: list = []


# ---------------------------------------------------
# PDF / DOCX EXTRACTION
# ---------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")  # Faster + cleaner
    return text


def extract_text_from_docx(file_bytes: bytes) -> str:
    file_stream = io.BytesIO(file_bytes)
    document = docx.Document(file_stream)
    return "\n".join(p.text for p in document.paragraphs)


# ---------------------------------------------------
# ⚡ 2-STEP OPTIMIZED AI ANALYSIS (SUPER FAST)
# ---------------------------------------------------

def analyze_with_ai(resume_text: str) -> AnalysisResult:
    # --------------------------
    # STEP 1 — Fast Summarization
    # --------------------------
    summary_prompt = f"""
Summarize this resume into clean structured points.
Keep it short but meaningful.

Resume:
\"\"\"{resume_text}\"\"\"
"""

    summary_response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": summary_prompt},
        ]
    )

    short_resume = summary_response.choices[0].message.content

    # --------------------------
    # STEP 2 — Actual Analysis (10× faster now)
    # --------------------------
    analysis_prompt = f"""
You are a professional resume reviewer.

Analyze this summarized resume and return STRICT JSON with:
- score (0-100)
- skills
- summary
- strengths
- weaknesses
- suggestions
- improvedResume

Summarized Resume:
\"\"\"{short_resume}\"\"\"
"""

    final_response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are an expert resume analyzer."},
            {"role": "user", "content": analysis_prompt},
        ],
        response_format={"type": "json_object"}
    )

    data = json.loads(final_response.choices[0].message.content)
    return AnalysisResult(**data)


# ---------------------------------------------------
# API — Analyze Resume
# ---------------------------------------------------

@app.post("/analyze-resume", response_model=AnalysisResult)
async def analyze_resume(file: UploadFile = File(...)):
    file_bytes = await file.read()
    fname = file.filename.lower()

    if fname.endswith(".pdf"):
        resume_text = extract_text_from_pdf(file_bytes)
    elif fname.endswith(".docx"):
        resume_text = extract_text_from_docx(file_bytes)
    else:
        return {"error": "Only PDF or DOCX files supported"}

    if not resume_text.strip():
        return {"error": "Could not extract text"}

    return analyze_with_ai(resume_text)


# ---------------------------------------------------
# API — Download Premium PDF
# ---------------------------------------------------

@app.post("/download-report")
async def download_report(data: AnalysisResult):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=45,
        rightMargin=45,
        topMargin=70,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()

    # Premium Styles
    title_style = ParagraphStyle(
        "title_style",
        parent=styles["Title"],
        fontSize=26,
        leading=32,
        textColor="#C9A227",
        alignment=1,
    )

    header_style = ParagraphStyle(
        "header_style",
        parent=styles["Heading2"],
        fontSize=18,
        textColor="#2E2E2E",
    )

    body_style = ParagraphStyle(
        "body_style",
        parent=styles["BodyText"],
        fontSize=12,
        leading=16,
        textColor="#333",
    )

    bullet_style = ParagraphStyle(
        "bullet_style",
        parent=styles["BodyText"],
        fontSize=12,
        leading=16,
        leftIndent=15,
    )

    # PDF BODY
    elements = []

    elements.append(Paragraph("Resume Analysis Report", title_style))
    elements.append(Spacer(1, 0.3 * inch))

    # Score Panel
    score_panel = [
        [Paragraph(f"<b>Resume Score: {data.score}/100</b>", header_style)]
    ]

    table = Table(score_panel, colWidths=[6 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.Color(0.95, 0.88, 0.55)),
        ("BOX", (0, 0), (-1, -1), 2, colors.Color(0.8, 0.65, 0.15)),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.25 * inch))

    # Summary
    elements.append(Paragraph("Professional Summary", header_style))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Paragraph(data.summary.replace("\n", "<br/>"), body_style))
    elements.append(Spacer(1, 0.25 * inch))

    # Strengths
    if data.strengths:
        elements.append(Paragraph("Key Strengths", header_style))
        for s in data.strengths:
            elements.append(Paragraph(f"• {s}", bullet_style))
        elements.append(Spacer(1, 0.25 * inch))

    # Weaknesses
    elements.append(Paragraph("Areas to Improve", header_style))
    for w in data.weaknesses:
        elements.append(Paragraph(f"• {w}", bullet_style))
    elements.append(Spacer(1, 0.25 * inch))

    # Skills List
    elements.append(Paragraph("Detected Skills", header_style))
    for skill in data.skills:
        elements.append(Paragraph(f"• {skill}", bullet_style))
    elements.append(Spacer(1, 0.3 * inch))

    # Suggestions
    elements.append(Paragraph("Suggestions", header_style))
    for sug in data.suggestions:
        elements.append(Paragraph(f"• {sug}", bullet_style))
    elements.append(Spacer(1, 0.3 * inch))

    # New Page — Improved Resume
    elements.append(PageBreak())
    elements.append(Paragraph("AI-Optimized Resume", title_style))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(data.improvedResume.replace("\n", "<br/>"), body_style))

    doc.build(elements)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=UltraPremium_Resume_Report.pdf"},
    )


# ---------------------------------------------------
# ROOT ROUTE
# ---------------------------------------------------

@app.get("/")
def home():
    return {"message": "Resume AI Backend Running Successfully!"}
