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

    buffer = io.BytesIO()

    # Document settings + custom margins
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=45,
        rightMargin=45,
        topMargin=70,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    # -----------------------------
    #    CUSTOM ULTRA PREMIUM STYLES
    # -----------------------------

    title_style = ParagraphStyle(
        "title_style",
        parent=styles["Title"],
        fontSize=28,
        leading=34,
        textColor="#C9A227",
        alignment=1,  # center
    )

    header_style = ParagraphStyle(
        "header_style",
        parent=styles["Heading2"],
        fontSize=18,
        leading=22,
        textColor="#2E2E2E",
    )

    subheader_style = ParagraphStyle(
        "subheader_style",
        parent=styles["Heading3"],
        fontSize=14,
        leading=18,
        textColor="#444",
    )

    body_style = ParagraphStyle(
        "body_style",
        parent=styles["BodyText"],
        fontSize=11.5,
        leading=16,
        textColor="#333",
    )

    bullet_style = ParagraphStyle(
        "bullet_style",
        parent=styles["BodyText"],
        fontSize=11.5,
        leading=16,
        leftIndent=15
    )

    highlight_box_style = ParagraphStyle(
        "highlight",
        parent=styles["BodyText"],
        backColor="#FFF8E1",
        borderColor="#C9A227",
        borderWidth=1,
        borderPadding=8,
        fontSize=12,
        leading=18,
        spaceAfter=12,
    )

    # -----------------------------
    # PAGE HEADER + TITLE
    # -----------------------------

    elements = []

    elements.append(Paragraph("Resume Analysis Report", title_style))
    elements.append(Spacer(1, 0.35 * inch))

    # -----------------------------
    # LARGE SCORE PANEL
    # -----------------------------

    score_panel = [
        [Paragraph(f"<b>Your Resume Score: {data.score}/100</b>", header_style)]
    ]

    table = Table(
        score_panel,
        colWidths=[6.2 * inch]
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.Color(0.95, 0.88, 0.55)),
        ("BOX", (0, 0), (-1, -1), 2, colors.Color(0.8, 0.65, 0.15)),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))

    # Divider
    elements.append(Paragraph("<para alignment='center'><font color='#C9A227'>────────────────────────────────────────</font></para>", body_style))
    elements.append(Spacer(1, 0.2 * inch))

    # -----------------------------
    # SUMMARY BLOCK
    # -----------------------------
    elements.append(Paragraph("Professional Summary", header_style))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Paragraph(data.summary.replace("\n", "<br/>"), body_style))
    elements.append(Spacer(1, 0.25 * inch))

    # -----------------------------
    # STRENGTHS (in gold box)
    # -----------------------------
    if data.strengths:
        elements.append(Paragraph("Key Strengths", header_style))
        elements.append(Spacer(1, 0.15 * inch))

        for s in data.strengths:
            elements.append(Paragraph(f"• {s}", bullet_style))

        elements.append(Spacer(1, 0.3 * inch))

    # -----------------------------
    # WEAKNESSES (in red box)
    # -----------------------------
    elements.append(Paragraph("Areas for Improvement", header_style))
    elements.append(Spacer(1, 0.15 * inch))

    for w in data.weaknesses:
        elements.append(Paragraph(f"• {w}", bullet_style))

    elements.append(Spacer(1, 0.3 * inch))

    # -----------------------------
    # SKILLS – Skill Badge Table
    # -----------------------------
    elements.append(Paragraph("Detected Skills", header_style))
    elements.append(Spacer(1, 0.15 * inch))

    skill_rows = []
    row = []

    for i, skill in enumerate(data.skills):
        badge = Paragraph(
            f"<para alignment='center'><b>{skill}</b></para>",
            ParagraphStyle(
                "badge",
                backColor="#EFEFEF",
                borderColor="#C9A227",
                borderWidth=1,
                borderRadius=5,
                alignment=1,
                padding=4,
                leading=14,
            )
        )
        row.append(badge)

        if len(row) == 3:
            skill_rows.append(row)
            row = []

    if row:
        skill_rows.append(row)

    skill_table = Table(skill_rows, colWidths=[2 * inch])
    skill_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    elements.append(skill_table)
    elements.append(Spacer(1, 0.3 * inch))

    # -----------------------------
    # SUGGESTIONS
    # -----------------------------
    elements.append(Paragraph("Suggestions for Improvement", header_style))
    elements.append(Spacer(1, 0.15 * inch))

    for s in data.suggestions:
        elements.append(Paragraph(f"• {s}", bullet_style))

    elements.append(Spacer(1, 0.4 * inch))

    # New Page for Improved Resume
    elements.append(PageBreak())

    # -----------------------------
    # IMPROVED RESUME (premium layout)
    # -----------------------------
    elements.append(Paragraph("AI-Optimized Resume", title_style))
    elements.append(Spacer(1, 0.3 * inch))

    improved_resume_text = data.improvedResume.replace("\n", "<br/>")
    elements.append(Paragraph(improved_resume_text, body_style))

    # -----------------------------
    # BUILD DOCUMENT
    # -----------------------------
    doc.build(elements)

    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=UltraPremium_Resume_Report.pdf"},
    )

# ---------------------------
# ROOT ROUTE
# ---------------------------

@app.get("/")
def home():
    return {"message": "Resume AI Backend Running Successfully!"}
