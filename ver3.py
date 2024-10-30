import pdfplumber
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os

# Load a pretrained NLP model (for English)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Define skill sets for various job profiles
job_profiles = {
    "Software Developer": ["Python", "Java", "C++", "Git", "Data Structures", "Algorithms"],
    "AI Engineer": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "NLP"],
    "SQL Developer": ["SQL", "Database Management", "ETL", "Data Warehousing", "Oracle", "MySQL"],
    "Data Scientist": ["Python", "R", "Statistics", "Data Analysis", "Machine Learning", "Pandas"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Git"],
    "DevOps Engineer": ["CI/CD", "Docker", "Kubernetes", "AWS", "Linux", "Shell Scripting"],
    "Mobile App Developer": ["Java", "Kotlin", "Swift", "Android", "iOS", "Flutter"],
    "Cybersecurity Analyst": ["Networking", "Firewalls", "Penetration Testing", "Cryptography", "Linux", "SIEM"]
}

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

    # Education and Experience Extraction
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if "education" in sent_text:
            entities["education"].append(sent.text.strip())
        elif "experience" in sent_text:
            entities["experience"].append(sent.text.strip())
    
    # Skill Extraction - matching with job profile skills
    all_skills = set([skill.lower() for skills in job_profiles.values() for skill in skills])
    found_skills = set(re.findall(r'\b\w+\b', text.lower()))
    matched_skills = list(all_skills.intersection(found_skills))
    entities["skills"].extend(matched_skills)
    
    return entities

def detect_job_profile(candidate_skills):
    """Detects the most suitable job profile for the candidate based on skills."""
    profile_scores = {profile: 0 for profile in job_profiles.keys()}
    
    for profile, required_skills in job_profiles.items():
        profile_scores[profile] = sum(1 for skill in required_skills if skill.lower() in candidate_skills)
    
    best_profile = max(profile_scores, key=profile_scores.get) if any(profile_scores.values()) else None
    return best_profile, profile_scores

def analyze_resume(text):
    """Analyzes the resume to find strengths, weaknesses, and suggests a job profile."""
    entities = extract_entities(text)
    
    # Flatten skills list and standardize for matching
    candidate_skills = set(skill.lower() for skill in entities["skills"])  # Standardized skills for matching

    best_profile, profile_scores = detect_job_profile(candidate_skills)
    
    # Identify missing skills for the detected profile
    required_skills = job_profiles.get(best_profile, [])
    missing_skills = [skill for skill in required_skills if skill.lower() not in candidate_skills]

    strengths = {
        "skills": list(candidate_skills),
        "education_level": "Bachelor's" if any("bachelor" in edu.lower() for edu in entities["education"]) else None,
    }
    weaknesses = {"missing_skills": missing_skills}
    
    return entities, strengths, weaknesses, best_profile, profile_scores

def save_visualization(skills_count, profile_scores, best_profile, strengths, weaknesses):
    """Generates and saves visualizations as images."""
    
    # Skills visualization
    if skills_count:
        skills, counts = zip(*skills_count.items())
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(skills), y=list(counts), palette="viridis")
        plt.title("Skills Mentioned in Resume")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("skills_visualization.png")
        plt.close()
    else:
        print("No skills found to visualize.")

    # Profile match visualization
    if any(profile_scores.values()):
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(profile_scores.keys()), y=list(profile_scores.values()), palette="coolwarm")
        plt.title("Profile Match Based on Skills")
        plt.xticks(rotation=45)
        plt.xlabel("Job Profiles")
        plt.ylabel("Matching Skill Count")
        plt.tight_layout()
        plt.savefig("profile_match_visualization.png")
        plt.close()
    else:
        print("No profile matches found for visualization.")

    # Skills analysis visualization
    missing_skills_count = len(weaknesses["missing_skills"])
    if missing_skills_count > 0 or strengths["skills"]:
        plt.figure(figsize=(5, 5))
        plt.pie([missing_skills_count, len(strengths["skills"])],
                labels=['Missing Skills', 'Present Skills'], autopct='%1.1f%%', startangle=90)
        plt.title(f"Skills Analysis for Best Profile: {best_profile}")
        plt.tight_layout()
        plt.savefig("skills_analysis_visualization.png")
        plt.close()
    else:
        print("No missing or present skills found for visualization.")


def generate_report(entities, strengths, weaknesses, best_profile, profile_scores, job_description):
    """Generates a report summarizing the candidate's resume analysis."""
    report_content = f"Resume Analysis Report\n"
    report_content += f"{'='*25}\n\n"

    # Job Description
    report_content += f"Job Description:\n{job_description}\n\n"
    
    # Profile Summary
    report_content += f"Suggested Job Profile: {best_profile}\n\n"
    
    # Strengths
    report_content += "Strengths:\n"
    report_content += f"- Skills Mentioned: {', '.join(strengths['skills']) if strengths['skills'] else 'None found'}\n"
    report_content += f"- Education Level: {strengths['education_level'] or 'Not specified'}\n\n"
    
    # Weaknesses
    report_content += "Weaknesses:\n"
    report_content += f"- Missing Skills for {best_profile} Profile: {', '.join(weaknesses['missing_skills']) if weaknesses['missing_skills'] else 'None'}\n\n"

    # Profile Scores
    report_content += "Profile Scores (Skill Matches):\n"
    for profile, score in profile_scores.items():
        report_content += f"- {profile}: {score} matching skills\n"
    report_content += "\n"

    # Recommendations
    report_content += "Recommendations:\n"
    if weaknesses["missing_skills"]:
        report_content += f"- To improve your profile as a {best_profile}, consider gaining skills in: "
        report_content += f"{', '.join(weaknesses['missing_skills'])}.\n"
    else:
        report_content += "- No additional skills are missing for your suggested profile!\n"
    
    report_content += "- Continue building expertise in areas related to your suggested profile to improve competitiveness.\n"

    # Save the report to a text file
    with open("resume_analysis_report.txt", "w") as report_file:
        report_file.write(report_content)

    # Include the visualizations in the report
    report_content += "\nVisualizations:\n"
    report_content += "1. Skills Mentioned: skills_visualization.png\n"
    report_content += "2. Profile Match: profile_match_visualization.png\n"
    report_content += "3. Skills Analysis: skills_analysis_visualization.png\n"
    
    # Save the updated report to the text file
    with open("resume_analysis_report.txt", "a") as report_file:
        report_file.write(report_content)

# Main Function
def main(pdf_path, job_description):
    # Extract and analyze resume text
    text = extract_text_from_pdf(pdf_path)
    entities, strengths, weaknesses, best_profile, profile_scores = analyze_resume(text)
    
    # Print analysis
    print("\nEntities Extracted:\n", entities)
    print("\nStrengths:\n", strengths)
    print("\nWeaknesses:\n", weaknesses)
    print("\nSuggested Job Profile:\n", best_profile)
    print("\nProfile Scores:\n", profile_scores)
    
    # Skills visualization
    skills_count = Counter(entities["skills"])
    
    # Save visualizations
    save_visualization(skills_count, profile_scores, best_profile, strengths, weaknesses)
    
    # Generate report
    generate_report(entities, strengths, weaknesses, best_profile, profile_scores, job_description)

# Run the model on a sample PDF resume
pdf_path = "C:\\Users\\krish\\Desktop\\Resume ai\\Krishn-Sharma_resume.pdf"  # Path to resume PDF
job_description = """
We are looking for a Software Developer with the following skills:
- Proficiency in Python, Java, and SQL
- Strong understanding of Data Structures and Algorithms
- Experience with Git and version control
- Familiarity with web technologies like HTML, CSS, and JavaScript
- Knowledge of Machine Learning and data analysis
"""  # Sample job description
main(pdf_path, job_description)



