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
    print("Extracted Text:\n", text)  # Debugging line to show extracted text
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
        elif "experience" in sent_text or "work" in sent_text:
            entities["experience"].append(sent.text.strip())
    
    # Skill Extraction - matching with job profile skills
    found_skills = set(re.findall(r'\b\w+\b', text.lower()))
    all_skills = set([skill.lower() for skills in job_profiles.values() for skill in skills])
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
    candidate_skills = set(skill.lower() for skill in entities["skills"])

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

def visualize_resume(entities, strengths, weaknesses, best_profile, profile_scores):
    """Visualizes resume data using bar charts and pie charts."""
    # Skills visualization for detected profile
    skills_count = Counter(entities["skills"])
    
    if skills_count:
        skills, counts = zip(*skills_count.items())
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(skills), y=list(counts), palette="viridis")
        plt.title("Skills Mentioned in Resume")
        plt.xticks(rotation=45)
        plt.show()
    else:
        print("No skills found to visualize.")

    # Visualize profile match
    if any(profile_scores.values()):
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(profile_scores.keys()), y=list(profile_scores.values()), palette="coolwarm")
        plt.title("Profile Match Based on Skills")
        plt.xticks(rotation=45)
        plt.xlabel("Job Profiles")
        plt.ylabel("Matching Skill Count")
        plt.show()
    else:
        print("No profile matches found for visualization.")

    # Missing skills for best-fit profile
    missing_skills_count = len(weaknesses["missing_skills"])
    if missing_skills_count > 0 or strengths["skills"]:
        plt.figure(figsize=(5, 5))
        plt.pie([missing_skills_count, len(strengths["skills"])], labels=['Missing Skills', 'Present Skills'], autopct='%1.1f%%', startangle=90)
        plt.title(f"Skills Analysis for Best Profile: {best_profile}")
        plt.show()
    else:
        print("No missing or present skills found for visualization.")

def generate_report(entities, strengths, weaknesses, best_profile, profile_scores):
    """Generates a report summarizing the candidate's resume analysis."""
    report = f"Resume Analysis Report\n"
    report += f"{'='*25}\n\n"

    # Profile Summary
    report += f"Suggested Job Profile: {best_profile}\n\n"
    
    # Strengths
    report += "Strengths:\n"
    report += f"- Skills Mentioned: {', '.join(strengths['skills']) if strengths['skills'] else 'None found'}\n"
    report += f"- Education Level: {strengths['education_level'] or 'Not specified'}\n\n"
    
    # Weaknesses
    report += "Weaknesses:\n"
    report += f"- Missing Skills for {best_profile} Profile: {', '.join(weaknesses['missing_skills']) if weaknesses['missing_skills'] else 'None'}\n\n"

    # Profile Scores
    report += "Profile Scores (Skill Matches):\n"
    for profile, score in profile_scores.items():
        report += f"- {profile}: {score} matching skills\n"
    report += "\n"

    # Recommendations
    report += "Recommendations:\n"
    if weaknesses["missing_skills"]:
        report += f"- To improve your profile as a {best_profile}, consider gaining skills in: "
        report += f"{', '.join(weaknesses['missing_skills'])}.\n"
    else:
        report += "- No additional skills are missing for your suggested profile!\n"
    
    report += "- Continue building expertise in areas related to your suggested profile to improve competitiveness.\n"

    print(report)

# Main Function
def main(pdf_path):
    # Extract and analyze resume text
    text = extract_text_from_pdf(pdf_path)
    entities, strengths, weaknesses, best_profile, profile_scores = analyze_resume(text)
    
    # Print analysis
    print("\nEntities Extracted:\n", entities)
    print("\nStrengths:\n", strengths)
    print("\nWeaknesses:\n", weaknesses)
    print("\nSuggested Job Profile:\n", best_profile)
    print("\nProfile Scores:\n", profile_scores)
    
    # Visualize the resume data
    visualize_resume(entities, strengths, weaknesses, best_profile, profile_scores)
    
    # Generate report
    generate_report(entities, strengths, weaknesses, best_profile, profile_scores)

# Run the model on a sample PDF resume
pdf_path = "C:\\Users\\krish\\Desktop\\Resume ai\\Krishn-Sharma_resume.pdf"  # Path to resume PDF
main(pdf_path)
