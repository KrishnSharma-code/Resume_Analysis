import pdfplumber
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Load a pretrained NLP model (for English)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model is not downloaded, download it
    print("Downloading 'en_core_web_sm' model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() or ''  # Handle pages without text
    return text

def extract_entities(text):
    """Extracts entities like skills, experience, education, etc., from resume text."""
    doc = nlp(text)
    entities = {"education": [], "experience": [], "skills": []}
    for sent in doc.sents:
        if "education" in sent.text.lower():
            entities["education"].append(sent.text.strip())
        elif "experience" in sent.text.lower():
            entities["experience"].append(sent.text.strip())
        elif re.search(r"(?i)\b(skill|technolog|proficien|tool)\b", sent.text):
            entities["skills"].append(sent.text.strip())
    
    # Debug print to check extracted entities
    print("\nExtracted Entities:", entities)
    
    return entities

def analyze_resume(text):
    """Analyzes the resume to find strengths and weaknesses."""
    entities = extract_entities(text)
    
    # Define required skills for analysis
    required_skills = ["Python", "Java", "SQL", "Machine Learning", "Data Science", "JavaScript", "HTML", "CSS", "Git", "C++"]
    skills_mentioned = set(skill for skill in required_skills if skill.lower() in text.lower())
    
    strengths = {
        "skills": list(skills_mentioned),
        "education_level": "Bachelor's" if any("bachelor" in edu.lower() for edu in entities["education"]) else None,
    }
    
    weaknesses = {
        "missing_skills": [skill for skill in required_skills if skill not in strengths["skills"]],
    }
    
    # Debug print to check strengths and weaknesses
    print("\nStrengths:", strengths)
    print("\nWeaknesses:", weaknesses)
    
    return entities, strengths, weaknesses

def visualize_resume(entities, strengths, weaknesses):
    """Visualizes resume data using bar charts and pie charts."""
    # Skills visualization
    skills_count = Counter(entities["skills"])
    
    # Handle case when no skills are found
    if skills_count:
        skills, counts = zip(*skills_count.items())
        
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(skills), y=list(counts), palette="viridis")
        plt.title("Skills Mentioned in Resume")
        plt.xticks(rotation=45)
        plt.show()
    else:
        print("No skills found to visualize.")

    # Experience visualization
    experience_count = len(entities["experience"])
    plt.figure(figsize=(5, 5))
    plt.pie([experience_count, max(1, 10 - experience_count)], labels=['Experience', 'Other'], autopct='%1.1f%%', startangle=90)
    plt.title("Experience Level")
    plt.show()

    # Weaknesses visualization
    missing_skills_count = len(weaknesses["missing_skills"])
    present_skills_count = len(strengths["skills"])
    plt.figure(figsize=(5, 5))
    plt.pie([missing_skills_count, present_skills_count], labels=['Missing Skills', 'Present Skills'], autopct='%1.1f%%', startangle=90)
    plt.title("Skills Analysis")
    plt.show()

# Main Function
def main(pdf_path):
    # Extract and analyze resume text
    text = extract_text_from_pdf(pdf_path)
    entities, strengths, weaknesses = analyze_resume(text)
    
    # Print analysis
    print("\nEntities Extracted:\n", entities)
    print("\nStrengths:\n", strengths)
    print("\nWeaknesses:\n", weaknesses)
    
    # Visualize the resume data
    visualize_resume(entities, strengths, weaknesses)

# Run the model on a sample PDF resume
pdf_path = "C:\\Users\\krish\\Desktop\\Resume ai\\Krishn-Sharma_resume.pdf"  # Path to resume PDF
main(pdf_path)
