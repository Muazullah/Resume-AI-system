import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PyPDF2 import PdfReader
from rapidfuzz import fuzz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.platypus import SimpleDocTemplate, Table, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_csv("final_merged_dataset2.csv")
df.columns = df.columns.str.strip()

TEXT_COL = "Resume"
LABEL_COL = "Category"

# -------------------------------
# MODEL
# -------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df[TEXT_COL])

model = MultinomialNB()
model.fit(X, df[LABEL_COL])

# -------------------------------
# SKILLS & WEIGHTS
# -------------------------------
skills = [
    "python", "machine learning", "sql", "deep learning",
    "tensorflow", "pytorch", "html", "css", "javascript",
    "react", "networking", "ccna", "routing"
]

weights = {
    "python": 10,
    "machine learning": 20,
    "deep learning": 20,
    "react": 15,
    "sql": 10,
    "tensorflow": 15,
    "pytorch": 15
}

# -------------------------------
# PDF READ
# -------------------------------
def read_pdf(file):
    if file is None:
        return ""
    try:
        reader = PdfReader(file.name)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t
        return text.lower().strip()
    except Exception:
        return ""

# -------------------------------
# SKILL DETECTION
# -------------------------------
def detect_skills(text):
    found = set()
    words = set(text.split())
    for skill in skills:
        if skill in text:
            found.add(skill)
        else:
            for w in words:
                if fuzz.partial_ratio(skill, w) > 90:
                    found.add(skill)
                    break
    return list(found)

# -------------------------------
# SINGLE RESUME ANALYSIS
# -------------------------------
def analyze(file):
    text = read_pdf(file)
    if not text:
        return "❌ Could not read the PDF. Please upload a valid resume."

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    conf = max(model.predict_proba(vec)[0]) * 100
    found = detect_skills(text)
    score = sum(weights.get(s, 5) for s in found)
    score = min(max(score, 0), 100)
    sim = max(cosine_similarity(vec, X)[0]) * 100

    skill_list = ", ".join(f"`{s}`" for s in found) if found else "None detected"

    return f"""### 🧠 Analysis Result

| Metric | Value |
|---|---|
| **Domain** | {pred} |
| **Confidence** | {conf:.1f}% |
| **Skill Score** | {score} / 100 |
| **Dataset Similarity** | {sim:.1f}% |

**🧰 Detected Skills:** {skill_list}
"""

# -------------------------------
# MULTI RESUME COMPARISON
# -------------------------------
def multi(files):
    if not files:
        return pd.DataFrame(columns=["File", "Score", "Domain"])

    results = []
    for f in files:
        t = read_pdf(f)
        if not t:
            continue
        vec = vectorizer.transform([t])
        sim = max(cosine_similarity(vec, X)[0]) * 100
        pred = model.predict(vec)[0]
        results.append({
            "File": os.path.basename(f.name),
            "Domain": pred,
            "Score": round(sim, 2)
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("Score", ascending=False).reset_index(drop=True)
    return result_df

# -------------------------------
# JD MATCH
# -------------------------------
def jd_match(file, jd):
    t = read_pdf(file)
    if not t:
        return "❌ Could not read the resume PDF."
    if not jd or not jd.strip():
        return "❌ Please enter a Job Description."

    v1 = vectorizer.transform([t])
    v2 = vectorizer.transform([jd.lower()])
    sim = cosine_similarity(v1, v2)[0][0] * 100

    sk_resume = detect_skills(t)
    sk_jd = detect_skills(jd.lower())
    matched = set(sk_resume) & set(sk_jd)
    missing = set(sk_jd) - set(sk_resume)

    matched_str = ", ".join(f"`{s}`" for s in matched) if matched else "None"
    missing_str = ", ".join(f"`{s}`" for s in missing) if missing else "None"

    return f"""### 🎯 JD Match Report

| Metric | Value |
|---|---|
| **Similarity Score** | {sim:.1f}% |
| **Skills Matched** | {len(matched)} / {len(sk_jd)} |

**✅ Matched Skills:** {matched_str}

**⚠️ Missing Skills:** {missing_str}
"""

# -------------------------------
# HR SCREENING — returns (df, state)
# -------------------------------
def hr_screen(files, jd, state):
    if not files:
        return pd.DataFrame(columns=["Candidate", "Domain", "JD Match %", "Skill Score", "Final Score"]), state
    if not jd or not jd.strip():
        return pd.DataFrame(columns=["Candidate", "Domain", "JD Match %", "Skill Score", "Final Score"]), state

    jd_lower = jd.lower()
    jd_vec = vectorizer.transform([jd_lower])
    jd_skills = detect_skills(jd_lower)

    results = []
    for f in files:
        t = read_pdf(f)
        if not t:
            continue
        vec = vectorizer.transform([t])
        sim = cosine_similarity(vec, jd_vec)[0][0] * 100
        pred = model.predict(vec)[0]
        sk = detect_skills(t)
        matched = set(jd_skills) & set(sk)
        skill_score = (len(matched) / len(jd_skills) * 100) if len(jd_skills) > 0 else 0
        final = 0.7 * sim + 0.3 * skill_score

        results.append({
            "Candidate": os.path.basename(f.name),
            "Domain": pred,
            "JD Match %": round(sim, 2),
            "Skill Score": round(skill_score, 2),
            "Final Score": round(final, 2)
        })

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values("Final Score", ascending=False).reset_index(drop=True)

    # Save to state (serializable dict)
    new_state = df_res.to_dict("records") if not df_res.empty else []
    return df_res, new_state

# -------------------------------
# DASHBOARD — reads from state
# -------------------------------
def dashboard(state):
    if not state:
        return "⚠️ Please run **HR Screening** first to generate data.", None, None

    hr_data = pd.DataFrame(state)

    top = hr_data.iloc[0]
    avg = hr_data["Final Score"].mean()

    # Chart 1: Score Distribution
    fig1 = px.histogram(
        hr_data, x="Final Score", nbins=10,
        title="Score Distribution",
        color_discrete_sequence=["#00d4ff"]
    )
    fig1.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        title_font_size=16
    )

    # Chart 2: Candidate Ranking
    fig2 = px.bar(
        hr_data, x="Candidate", y="Final Score",
        title="Candidate Ranking",
        color="Final Score",
        color_continuous_scale="Blues"
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        title_font_size=16,
        xaxis_tickangle=-30
    )

    summary = f"""### 📊 Screening Summary

| | |
|---|---|
| 🏆 **Top Candidate** | {top['Candidate']} |
| 🎯 **Top Score** | {top['Final Score']} |
| 📈 **Average Score** | {avg:.2f} |
| 👥 **Total Screened** | {len(hr_data)} |
"""
    return summary, fig1, fig2

# -------------------------------
# PDF EXPORT — reads from state
# -------------------------------
def export_pdf(state):
    if not state:
        return None

    hr_data = pd.DataFrame(state)
    file_path = "hr_report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("HR Screening Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    cols = list(hr_data.columns)
    data = [cols] + [[str(hr_data[c].iloc[i]) for c in cols] for i in range(len(hr_data))]

    table = Table(data, repeatRows=1)
    table.setStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f4f8fc"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ccd9e8")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ])

    elements.append(table)
    doc.build(elements)
    return file_path

# -------------------------------
# PROFESSIONAL DARK CSS
# -------------------------------
css = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    background: #080d16 !important;
    color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, #0f1e38 0%, #0d1a2e 100%);
    border-bottom: 1px solid #1e3a5f;
    margin-bottom: 1.5rem;
}
.app-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.app-subtitle {
    color: #64748b;
    font-size: 0.9rem;
    margin-top: 0.25rem;
}

/* Tabs */
.tab-nav {
    background: #0d1a2e !important;
    border-bottom: 1px solid #1e3a5f !important;
}
.tab-nav button {
    color: #64748b !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.75rem 1.2rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
}
.tab-nav button.selected {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
    background: transparent !important;
}
.tab-nav button:hover {
    color: #94a3b8 !important;
    background: rgba(56,189,248,0.05) !important;
}

/* Panels */
.gradio-tabitem {
    background: #0d1a2e !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    margin-top: 0.5rem !important;
}

/* Inputs */
input, textarea, .gr-box {
    background: #0f1f38 !important;
    border: 1px solid #1e3a5f !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
input:focus, textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
    outline: none !important;
}
label span {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* File upload */
.upload-container, .gr-file-upload {
    background: #0f1f38 !important;
    border: 2px dashed #1e3a5f !important;
    border-radius: 10px !important;
    transition: border-color 0.2s !important;
}
.upload-container:hover {
    border-color: #38bdf8 !important;
}

/* Buttons */
button.primary, .gr-button-primary, button[variant="primary"] {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.5rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.1s !important;
    box-shadow: 0 4px 15px rgba(14,165,233,0.3) !important;
}
button.primary:hover, .gr-button-primary:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
button.secondary, .gr-button-secondary {
    background: transparent !important;
    border: 1px solid #1e3a5f !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
}

/* Dataframe */
.gr-dataframe, table {
    background: #0f1f38 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
table th {
    background: #1e3a5f !important;
    color: #38bdf8 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    padding: 0.75rem 1rem !important;
    font-weight: 600 !important;
}
table td {
    color: #cbd5e1 !important;
    border-color: #1e3a5f !important;
    padding: 0.65rem 1rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}
table tr:hover td {
    background: rgba(56,189,248,0.05) !important;
}

/* Markdown output */
.gr-markdown {
    background: #0f1f38 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
    padding: 1.25rem !important;
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
}
.gr-markdown h3 {
    color: #38bdf8 !important;
    font-size: 1rem !important;
    margin-bottom: 0.75rem !important;
    border-bottom: 1px solid #1e3a5f !important;
    padding-bottom: 0.5rem !important;
}
.gr-markdown table { margin: 0.75rem 0 !important; }
.gr-markdown code {
    background: rgba(56,189,248,0.1) !important;
    color: #38bdf8 !important;
    padding: 0.15rem 0.4rem !important;
    border-radius: 4px !important;
    font-size: 0.8rem !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #080d16; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
"""

# -------------------------------
# UI
# -------------------------------
with gr.Blocks(css=css, title="Resume AI System") as demo:

    # Shared state for HR screening results
    hr_state = gr.State([])

    gr.HTML("""
    <div class="app-header">
        <div class="app-title">⚡ Resume AI System</div>
        <div class="app-subtitle">Intelligent Resume Screening &amp; Analytics Platform</div>
    </div>
    """)

    with gr.Tabs():

        # ── TAB 1: Single Analyze ──────────────────────────
        with gr.Tab("🔍 Analyze Resume"):
            gr.Markdown("Upload a resume PDF to get domain classification, confidence score, and skill detection.")
            with gr.Row():
                with gr.Column(scale=1):
                    f1 = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
                    btn1 = gr.Button("▶ Analyze", variant="primary")
                with gr.Column(scale=2):
                    o1 = gr.Markdown(label="Analysis Output")
            btn1.click(analyze, inputs=f1, outputs=o1)

        # ── TAB 2: Multi Resume ────────────────────────────
        with gr.Tab("📂 Batch Compare"):
            gr.Markdown("Upload multiple resumes to compare them side-by-side by similarity score.")
            f2 = gr.File(label="Upload Resumes (PDF)", file_count="multiple", file_types=[".pdf"])
            btn2 = gr.Button("▶ Compare All", variant="primary")
            o2 = gr.Dataframe(label="Comparison Results", interactive=False)
            btn2.click(multi, inputs=f2, outputs=o2)

        # ── TAB 3: JD Match ───────────────────────────────
        with gr.Tab("🎯 JD Match"):
            gr.Markdown("Check how well a resume matches a specific job description.")
            with gr.Row():
                with gr.Column(scale=1):
                    f3 = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
                    jd = gr.Textbox(label="Paste Job Description", lines=6, placeholder="Enter the full job description here...")
                    btn3 = gr.Button("▶ Match Resume", variant="primary")
                with gr.Column(scale=1):
                    o3 = gr.Markdown(label="Match Report")
            btn3.click(jd_match, inputs=[f3, jd], outputs=o3)

        # ── TAB 4: HR Screening ───────────────────────────
        with gr.Tab("🏢 HR Screening"):
            gr.Markdown("Screen multiple candidates against a job description. Results are saved for the Dashboard.")
            with gr.Row():
                with gr.Column(scale=1):
                    jd2 = gr.Textbox(label="Job Description", lines=6, placeholder="Paste the job description here...")
                    f4 = gr.File(label="Upload Candidate Resumes (PDF)", file_count="multiple", file_types=[".pdf"])
                    btn4 = gr.Button("▶ Screen Candidates", variant="primary")
                with gr.Column(scale=2):
                    o4 = gr.Dataframe(label="Screening Results", interactive=False)

            btn4.click(
                hr_screen,
                inputs=[f4, jd2, hr_state],
                outputs=[o4, hr_state]
            )

        # ── TAB 5: Dashboard ──────────────────────────────
        with gr.Tab("📊 Live Dashboard"):
            gr.Markdown("Visual analytics from the latest HR Screening run.")
            btn5 = gr.Button("🔄 Refresh Dashboard", variant="primary")
            dash_summary = gr.Markdown()
            with gr.Row():
                chart1 = gr.Plot(label="Score Distribution")
                chart2 = gr.Plot(label="Candidate Ranking")

            btn5.click(
                dashboard,
                inputs=[hr_state],
                outputs=[dash_summary, chart1, chart2]
            )

        # ── TAB 6: Export PDF ─────────────────────────────
        with gr.Tab("📄 Export Report"):
            gr.Markdown("Generate a formatted PDF report from the latest HR Screening results.")
            btn6 = gr.Button("📥 Generate PDF Report", variant="primary")
            file_out = gr.File(label="Download Report")

            btn6.click(
                export_pdf,
                inputs=[hr_state],
                outputs=file_out
            )

demo.launch()