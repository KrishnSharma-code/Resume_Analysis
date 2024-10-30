import pdfplumber
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
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

def save_visualizations(entities, strengths, weaknesses, best_profile, profile_scores, output_dir):
    """Saves visualizations as images."""
    # Skills visualization for detected profile
    skills_count = Counter(entities["skills"])
    
    if skills_count:
        skills, counts = zip(*skills_count.items())
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(skills), y=list(counts), palette="viridis")
        plt.title("Skills Mentioned in Resume")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "skills_visualization.png"))
        plt.close()
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
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "profile_match_visualization.png"))
        plt.close()
    else:
        print("No profile matches found for visualization.")

    # Missing skills for best-fit profile
    missing_skills_count = len(weaknesses["missing_skills"])
    if missing_skills_count > 0 or strengths["skills"]:
        plt.figure(figsize=(5, 5))
        plt.pie([missing_skills_count, len(strengths["skills"])], labels=['Missing Skills', 'Present Skills'], autopct='%1.1f%%', startangle=90)
        plt.title(f"Skills Analysis for Best Profile: {best_profile}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "skills_analysis_visualization.png"))
        plt.close()
    else:
        print("No missing or present skills found for visualization.")

def generate_report(entities, strengths, weaknesses, best_profile, profile_scores, output_dir):
    """Generates a report summarizing the candidate's resume analysis."""
    report_path = os.path.join(output_dir, "resume_analysis_report.pdf")
    c = canvas.Canvas(report_path, pagesize=letter)

    def add_text_with_new_page(text, y_position):
        """Add text to canvas and handle new page if needed."""
        for line in text.split('\n'):
            if y_position < 1 * inch:
                c.showPage()  # Create a new page
                y_position = 10 * inch  # Reset y position for the new page
            c.drawString(1 * inch, y_position, line)
            y_position -= 0.2 * inch
        return y_position

    # Report Header
    y_position = 10 * inch
    c.drawString(1 * inch, y_position, "Resume Analysis Report")
    y_position -= 0.5 * inch
    c.drawString(1 * inch, y_position, "=" * 25)
    y_position -= 0.5 * inch

    # Profile Summary
    c.drawString(1 * inch, y_position, f"Suggested Job Profile: {best_profile}")
    y_position -= 0.5 * inch

    # Strengths
    y_position = add_text_with_new_page("Strengths:", y_position)
    y_position = add_text_with_new_page(f"- Skills Mentioned: {', '.join(strengths['skills']) if strengths['skills'] else 'None found'}", y_position)
    y_position = add_text_with_new_page(f"- Education Level: {strengths['education_level'] or 'Not specified'}", y_position)

    # Weaknesses
    y_position = add_text_with_new_page("Weaknesses:", y_position)
    y_position = add_text_with_new_page(f"- Missing Skills for {best_profile} Profile: {', '.join(weaknesses['missing_skills']) if weaknesses['missing_skills'] else 'None'}", y_position)

    # Profile Scores
    y_position = add_text_with_new_page("Profile Scores (Skill Matches):", y_position)
    for profile, score in profile_scores.items():
        y_position = add_text_with_new_page(f"- {profile}: {score} matching skills", y_position)

    # Recommendations
    y_position = add_text_with_new_page("Recommendations:", y_position)
    if weaknesses["missing_skills"]:
        y_position = add_text_with_new_page(f"- To improve your profile as a {best_profile}, consider gaining skills in: ", y_position)
        y_position = add_text_with_new_page(f"{', '.join(weaknesses['missing_skills'])}.", y_position)
    else:
        y_position = add_text_with_new_page("- No additional skills are missing for your suggested profile!", y_position)

    y_position = add_text_with_new_page("- Continue building expertise in areas related to your suggested profile to improve competitiveness.", y_position)

    # Add visualizations to the report after the text
    c.showPage()  # Start a new page for graphs
    c.drawString(1 * inch, 10 * inch, "Visualizations")
    y_position = 9.5 * inch

    # Skills Visualization
    c.drawImage(os.path.join(output_dir, "skills_visualization.png"), 1 * inch, y_position - 2 * inch, width=5 * inch, height=2.5 * inch)
    y_position -= 2.8 * inch  # Adjust position for next image
    if y_position < 1 * inch:  # Check if we need a new page
        c.showPage()
        y_position = 10 * inch
    c.drawString(1 * inch, y_position, "Skills Visualization")
    y_position -= 0.5 * inch

    # Profile Match Visualization
    c.drawImage(os.path.join(output_dir, "profile_match_visualization.png"), 1 * inch, y_position - 2 * inch, width=5 * inch, height=2.5 * inch)
    y_position -= 2.8 * inch  # Adjust position for next image
    if y_position < 1 * inch:  # Check if we need a new page
        c.showPage()
        y_position = 10 * inch
    c.drawString(1 * inch, y_position, "Profile Match Visualization")
    y_position -= 0.5 * inch

    # Skills Analysis Visualization
    c.drawImage(os.path.join(output_dir, "skills_analysis_visualization.png"), 1 * inch, y_position - 2 * inch, width=5 * inch, height=2.5 * inch)

    c.save()
    print(f"Report generated at: {report_path}")



# Main Function
def main(pdf_path):
    output_dir = "resume_analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract and analyze resume text
    text = extract_text_from_pdf(pdf_path)
    entities, strengths, weaknesses, best_profile, profile_scores = analyze_resume(text)
    
    # Print analysis
    print("\nEntities Extracted:\n", entities)
    print("\nStrengths:\n", strengths)
    print("\nWeaknesses:\n", weaknesses)
    print("\nSuggested Job Profile:\n", best_profile)
    print("\nProfile Scores:\n", profile_scores)
    
    # Save visualizations
    save_visualizations(entities, strengths, weaknesses, best_profile, profile_scores, output_dir)
    
    # Generate report
    generate_report(entities, strengths, weaknesses, best_profile, profile_scores, output_dir)

# Run the model on a sample PDF resume
pdf_path = "C:\\Users\\krish\\Desktop\\Resume ai\\Krishn-Sharma_resume.pdf"  # Path to resume PDF
main(pdf_path)
