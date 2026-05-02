After running, open the local URL (typically `http://127.0
```markdown
# ⚡ Resume AI System

An intelligent, full-stack **Resume Screening & Analytics Platform** built with Python and Gradio. This system leverages Natural Language Processing (NLP) to automate the recruitment process, offering deep insights into candidate profiles, job description matching, and batch HR processing.

## 🚀 Key Features

*   **🔍 Single Resume Analysis:** Classifies resumes into professional domains with a confidence score and detects specific technical skills using fuzzy matching.
*   **📂 Batch Compare:** Upload multiple resumes simultaneously to compare them based on their similarity to the internal training dataset.
*   **🎯 JD Match:** Compare a candidate's resume directly against a specific Job Description to identify matched and missing skills.
*   **🏢 HR Screening:** Specialized tool for recruiters to rank candidates against a JD using a weighted scoring algorithm (70% semantic similarity, 30% skill matching).
*   **📊 Live Dashboard:** Visual analytics powered by Plotly, featuring score distribution histograms and candidate ranking bar charts.
*   **📄 PDF Export:** Generate and download formatted PDF reports containing the results of the HR screening process.
*   **🎨 Premium UI:** A modern, dark-themed interface built with custom CSS for a professional user experience.

---

## 🛠️ Project Structure

```text
MLPROJECT/
├── app.py                      # Main application logic and Gradio UI
├── final_merged_dataset2.csv   # Dataset used for TF-IDF model training
├── requirements.txt            # Project dependencies
└── readme.md                   # Project documentation
