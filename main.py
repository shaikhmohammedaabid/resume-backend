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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from fastapi.responses import StreamingResponse


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
    Generates a premium PDF report with elegant formatting.
    """

    buffer = io.BytesIO()

    # PREMIUM PAGE SETTINGS
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=60,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    title_style.fontSize = 26
    title_style.textColor = "#C9A227"  # Gold color
    title_style.leading = 32

    heading_style = styles["Heading2"]
    heading_style.fontSize = 18
    heading_style.textColor = "#333333"

    subheading_style = styles["Heading3"]
    subheading_style.fontSize = 14
    subheading_style.textColor = "#555555"

    body = styles["BodyText"]
    body.fontName = "Helvetica"
    body.fontSize = 11
    body.leading = 14

    bullet_style = styles["Bullet"]
    bullet_style.fontSize = 11
    bullet_style.leading = 14

    elements = []

    # -------------------------
    # TITLE SECTION
    # -------------------------
    elements.append(Paragraph("<b>Resume Analysis Report</b>", title_style))
    elements.append(Spacer(1, 0.3 * inch))

    # -------------------------
    # SCORE SECTION BOX
    # -------------------------
    score_box = f"""
    <para alignment="center">
        <font size="18" color="#C9A227"><b>Resume Score: {data.score}/100</b></font><br/>
        <font size="12">A higher score means your resume is more job-ready.</font>
    </para>
    """

    elements.append(Paragraph(score_box, body))
    elements.append(Spacer(1, 0.3 * inch))

    # Divider Line
    elements.append(Paragraph("<para><font color='#C9A227'>────────────────────────────────────────────</font></para>", body))
    elements.append(Spacer(1, 0.2 * inch))

    # -------------------------
    # SUMMARY
    # -------------------------
    elements.append(Paragraph("<b>Professional Summary</b>", heading_style))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Paragraph(data.summary.replace("\n", "<br/>"), body))
    elements.append(Spacer(1, 0.25 * inch))

    # -------------------------
    # STRENGTHS
    # -------------------------
    if data.strengths:
        elements.append(Paragraph("<b>Key Strengths</b>", heading_style))
        elements.append(Spacer(1, 0.1 * inch))
        for s in data.strengths:
            elements.append(Paragraph(f"• {s}", bullet_style))
        elements.append(Spacer(1, 0.25 * inch))

    # -------------------------
    # WEAKNESSES
    # -------------------------
    elements.append(Paragraph("<b>Areas for Improvement</b>", heading_style))
    elements.append(Spacer(1, 0.1 * inch))
    for w in data.weaknesses:
        elements.append(Paragraph(f"• {w}", bullet_style))
    elements.append(Spacer(1, 0.25 * inch))

    # -------------------------
    # SKILLS
    # -------------------------
    elements.append(Paragraph("<b>Detected Skills</b>", heading_style))
    elements.append(Spacer(1, 0.1 * inch))

    for skill in data.skills:
        elements.append(Paragraph(f"• {skill}", bullet_style))
    elements.append(Spacer(1, 0.25 * inch))

    # -------------------------
    # SUGGESTIONS
    # -------------------------
    elements.append(Paragraph("<b>Suggestions</b>", heading_style))
    elements.append(Spacer(1, 0.1 * inch))

    for s in data.suggestions:
        elements.append(Paragraph(f"• {s}", bullet_style))
    elements.append(Spacer(1, 0.25 * inch))

    # Divider Line
    elements.append(Paragraph("<para><font color='#C9A227'>────────────────────────────────────────────</font></para>", body))
    elements.append(Spacer(1, 0.25 * inch))

    # -------------------------
    # IMPROVED RESUME SECTION
    # -------------------------
    elements.append(Paragraph("<b>AI-Improved Resume</b>", heading_style))
    elements.append(Spacer(1, 0.2 * inch))

    improved = data.improvedResume.replace("\n", "<br/>")
    elements.append(Paragraph(improved, body))

    # -------------------------
    # BUILD PDF
    # -------------------------
    doc.build(elements)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=Resume_Analysis_Report.pdf"},
    )



# ---------------------------
# ROOT ROUTE
# ---------------------------

@app.get("/")
def home():
    return {"message": "Resume AI Backend Running Successfully!"}
