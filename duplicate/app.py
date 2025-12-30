import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import os
import re
import random
import joblib

# Try importing pypdf
try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: 'pypdf' not installed.")

app = Flask(__name__, template_folder='.')

# --- Configuration with Absolute Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'enhanced_career_dataset.csv')
MODEL_FILE = os.path.join(BASE_DIR, 'career_model_final.pkl') 

# Global Artifacts
model_artifacts = {
    'pipeline': None,
    'le_field': None,
    'le_career': None,
    'career_stats': None,
    'salary_stats': None
}

SKILL_KEYWORDS = {
    'Coding_Skills': ['python', 'java', 'c++', 'sql', 'javascript', 'html', 'css', 'programming', 'code', 'developer', 'react', 'node', 'aws'],
    'Communication_Skills': ['communication', 'verbal', 'written', 'presentation', 'public speaking', 'proposal', 'report', 'client'],
    'Problem_Solving_Skills': ['problem', 'solving', 'analysis', 'critical', 'thinking', 'strategy', 'troubleshoot', 'optimization'],
    'Teamwork_Skills': ['team', 'collaboration', 'group', 'partner', 'joint', 'cooperation', 'leadership', 'management'],
    'Analytical_Skills': ['analysis', 'data', 'metrics', 'statistics', 'research', 'evaluation', 'quantitative', 'reporting'],
    'Project_Management': ['project', 'manage', 'coordination', 'planning', 'agile', 'scrum', 'schedule', 'budget']
}

def initialize_system():
    """Checks for existing model file; loads it if present, otherwise trains new model."""
    global model_artifacts
    
    if os.path.exists(MODEL_FILE):
        print(f"✅ Found existing model at: {MODEL_FILE}")
        try:
            model_artifacts = joblib.load(MODEL_FILE)
            print("   Model loaded successfully!")
            return
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            print("   Falling back to training new model...")
    else:
        print(f"ℹ️  No existing model found at: {MODEL_FILE}")
    
    train_and_save_model()

def train_and_save_model():
    global model_artifacts
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: Dataset not found at {DATA_FILE}")
        return

    print("🚀 Starting training process...")
    df = pd.read_csv(DATA_FILE)

    # 1. Encoders
    le_field = LabelEncoder()
    df['Field_Encoded'] = le_field.fit_transform(df['Field'])
    le_career = LabelEncoder()
    df['Career_Encoded'] = le_career.fit_transform(df['Career'])

    # 2. Features
    feature_cols = [
        'GPA', 'Extracurricular_Activities', 'Internships', 'Projects', 
        'Leadership_Positions', 'Field_Specific_Courses', 'Research_Experience', 
        'Coding_Skills', 'Communication_Skills', 'Problem_Solving_Skills', 
        'Teamwork_Skills', 'Analytical_Skills', 'Presentation_Skills', 
        'Networking_Skills', 'Industry_Certifications', 'Field_Encoded'
    ]
    
    for col in feature_cols:
        if col not in df.columns: df[col] = 0

    X = df[feature_cols]
    y = df['Career_Encoded']

    # 3. Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42))
    ])
    pipeline.fit(X, y)
    
    # 4. Stats
    stats_cols = [c for c in feature_cols if c != 'Field_Encoded']
    career_stats = df.groupby('Career')[stats_cols].mean().to_dict('index')
    salary_stats = df.groupby('Career')['Salary_LPA'].agg(['min', 'max', 'mean']).to_dict('index')

    model_artifacts = {
        'pipeline': pipeline,
        'le_field': le_field,
        'le_career': le_career,
        'career_stats': career_stats,
        'salary_stats': salary_stats
    }
    
    try:
        joblib.dump(model_artifacts, MODEL_FILE)
        print(f"💾 Model trained and saved to: {MODEL_FILE}")
    except Exception as e:
        print(f"⚠️ Warning: Could not save model file. {e}")

def parse_pdf(file_stream):
    if not PDF_AVAILABLE: return {}
    try:
        reader = PdfReader(file_stream)
        text = ""
        for page in reader.pages: text += page.extract_text() + " "
        text = text.lower()
        extracted_data = {}

        for skill, keywords in SKILL_KEYWORDS.items():
            count = sum(text.count(k) for k in keywords)
            if count >= 6: score = 9
            elif count >= 4: score = 7
            elif count >= 2: score = 5
            elif count >= 1: score = 3
            else: score = 1
            if skill == 'Project_Management': extracted_data['Projects'] = score
            else: extracted_data[skill] = score

        gpa_match = re.search(r'gpa\s*[:=]?\s*(\d\.\d+)', text)
        if gpa_match:
            try: extracted_data['GPA'] = min(float(gpa_match.group(1)), 10.0)
            except: pass
        
        extracted_data['Leadership_Positions'] = 8 if 'lead' in text or 'president' in text else 2
        extracted_data['Research_Experience'] = 8 if 'research' in text or 'lab' in text else 2
        extracted_data['Internships'] = min(text.count('intern') * 3, 10)
        
        return extracted_data
    except Exception as e:
        print(f"Parsing Error: {e}")
        return {}

def get_safe_float(data, key, default=0.0):
    """Helper to safely retrieve and convert values to float."""
    try:
        val = data.get(key, default)
        return float(val)
    except (ValueError, TypeError):
        return float(default)

def generate_ai_summary(data, top_careers, field):
    """Generates a rule-based AI summary."""
    skills_map = {
        'Coding_Skills': 'Technical Coding',
        'Communication_Skills': 'Communication',
        'Problem_Solving_Skills': 'Complex Problem Solving',
        'Analytical_Skills': 'Data Analysis',
        'Teamwork_Skills': 'Collaboration',
        'Leadership_Positions': 'Leadership'
    }
    
    # Safely check skills using helper logic
    strong_skills = []
    for key, name in skills_map.items():
        if get_safe_float(data, key) >= 6:
            strong_skills.append(name)
    
    summary = f"Based on your profile in {field}, "
    
    gpa = get_safe_float(data, 'GPA')
    if gpa >= 8.0:
        summary += "your exceptional academic record combined with "
    elif gpa >= 6.0:
        summary += "your solid academic foundation and "
    else:
        summary += "your practical experience and "

    if strong_skills:
        summary += f"strong capabilities in {', '.join(strong_skills[:3])} "
    else:
        summary += "balanced skill set "

    summary += f"make you a prime candidate for a {top_careers[0]['role']}. "
    if len(top_careers) > 2:
        summary += f"Your profile also shows strong transferability to roles like {top_careers[1]['role']} and {top_careers[2]['role']}, "
        summary += "offering you diverse opportunities in the current job market."
    else:
        summary += "This role aligns perfectly with your current skill trajectory."

    return summary

@app.route('/')
def home(): return render_template('index.html')

@app.route('/parse_resume', methods=['POST'])
def handle_parse_resume():
    if 'resume' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['resume']
    if file.filename == '': return jsonify({'error': 'No file'}), 400
    return jsonify(parse_pdf(file))

@app.route('/predict', methods=['POST'])
def predict():
    if model_artifacts['pipeline'] is None: return jsonify({'error': 'Model not initialized'}), 500
    
    data = request.json
    pipeline = model_artifacts['pipeline']
    le_field = model_artifacts['le_field']
    le_career = model_artifacts['le_career']
    career_stats = model_artifacts['career_stats']
    salary_stats = model_artifacts['salary_stats']
    
    try: field_enc = le_field.transform([data.get('Field', 'General')])[0]
    except: field_enc = 0 

    # Use safe float conversion for all inputs
    input_vector = [
        get_safe_float(data, 'GPA', 5.0), 
        get_safe_float(data, 'Extracurricular_Activities'), 
        get_safe_float(data, 'Internships'), 
        get_safe_float(data, 'Projects'), 
        get_safe_float(data, 'Leadership_Positions'), 
        get_safe_float(data, 'Field_Specific_Courses'), 
        get_safe_float(data, 'Research_Experience'), 
        get_safe_float(data, 'Coding_Skills'), 
        get_safe_float(data, 'Communication_Skills'), 
        get_safe_float(data, 'Problem_Solving_Skills'), 
        get_safe_float(data, 'Teamwork_Skills'), 
        get_safe_float(data, 'Analytical_Skills'), 
        get_safe_float(data, 'Presentation_Skills'), 
        get_safe_float(data, 'Networking_Skills'), 
        get_safe_float(data, 'Industry_Certifications'), 
        field_enc
    ]
    
    # --- PROBABILITY PREDICTION ---
    probs = pipeline.predict_proba([input_vector])[0]
    
    top_n = 4
    top_indices = np.argsort(probs)[::-1][:top_n]
    
    top_careers_names = le_career.inverse_transform(top_indices)
    top_scores = probs[top_indices]
    
    top_careers = []
    for name, score in zip(top_careers_names, top_scores):
        top_careers.append({
            'role': name,
            'match': round(score * 100, 1)
        })

    primary_career = top_careers[0]['role']

    # --- STATISTICS ---
    avg_stats = career_stats.get(primary_career, {})
    user_stats_chart = [
        get_safe_float(data, 'Coding_Skills'), get_safe_float(data, 'Problem_Solving_Skills'), 
        get_safe_float(data, 'Communication_Skills'), get_safe_float(data, 'Analytical_Skills'), 
        get_safe_float(data, 'Teamwork_Skills'), get_safe_float(data, 'GPA')
    ]
    target_stats_chart = [
        avg_stats.get('Coding_Skills', 0), avg_stats.get('Problem_Solving_Skills', 0),
        avg_stats.get('Communication_Skills', 0), avg_stats.get('Analytical_Skills', 0),
        avg_stats.get('Teamwork_Skills', 0), avg_stats.get('GPA', 0)
    ]

    ai_summary = generate_ai_summary(data, top_careers, data.get('Field', 'General'))

    recs = []
    if get_safe_float(data, 'GPA') < avg_stats.get('GPA', 0) - 1:
        recs.append(f"Aim for a GPA of {avg_stats.get('GPA', 0):.1f}+")
    if get_safe_float(data, 'Coding_Skills') < avg_stats.get('Coding_Skills', 0) - 2:
        recs.append("Critical: Improve technical coding skills.")
    if get_safe_float(data, 'Internships') < 3: 
        recs.append("Secure at least 2-3 internships.")
    if not recs:
        recs.append("Profile match is excellent! Focus on salary negotiation.")

    random.seed(primary_career)
    years = [str(y) for y in range(2020, 2030)]
    hiring_trend = []
    base_hiring = random.randint(500, 2000)
    for _ in years:
        base_hiring += random.randint(-50, 200)
        hiring_trend.append(base_hiring)
    listing_trend = [int(h * 1.2 + random.randint(0, 50)) for h in hiring_trend]

    sal_info = salary_stats.get(primary_career, {'min': 10, 'mean': 25, 'max': 50})
    salary_trend_data = [
        sal_info['min'], (sal_info['min'] + sal_info['mean'])/2, 
        sal_info['mean'], (sal_info['mean'] + sal_info['max'])/2, sal_info['max']
    ]
    salary_labels = ["Entry Level", "Junior", "Mid Level", "Senior", "Lead/Principal"]
    scope_score = min(int((sal_info['mean'] / 80) * 100) + random.randint(0, 20), 100)

    return jsonify({
        'top_careers': top_careers,
        'ai_summary': ai_summary,
        'user_stats': user_stats_chart,
        'career_stats': target_stats_chart,
        'recommendations': recs,
        'years': years,
        'hiring_trend': hiring_trend,
        'listing_trend': listing_trend,
        'salary_labels': salary_labels,
        'salary_data': salary_trend_data,
        'scope_score': scope_score,
        'avg_salary': f"{sal_info['mean']:.2f} LPA"
    })

@app.route('/explore_career', methods=['POST'])
def explore_career():
    """Generates detailed stats for a specific career chosen by the user."""
    data = request.json
    target_career = data.get('target_career')
    user_inputs = data.get('user_data', {})
    match_score = data.get('match_score')

    career_stats = model_artifacts['career_stats']
    salary_stats = model_artifacts['salary_stats']
    
    # 1. Get Stats for the Target Career
    avg_stats = career_stats.get(target_career, {})
    
    # 2. Re-construct User Stats Vector for Charts using SAFE FLOAT
    user_stats_chart = [
        get_safe_float(user_inputs, 'Coding_Skills'), get_safe_float(user_inputs, 'Problem_Solving_Skills'), 
        get_safe_float(user_inputs, 'Communication_Skills'), get_safe_float(user_inputs, 'Analytical_Skills'), 
        get_safe_float(user_inputs, 'Teamwork_Skills'), get_safe_float(user_inputs, 'GPA')
    ]
    
    target_stats_chart = [
        avg_stats.get('Coding_Skills', 0), avg_stats.get('Problem_Solving_Skills', 0),
        avg_stats.get('Communication_Skills', 0), avg_stats.get('Analytical_Skills', 0),
        avg_stats.get('Teamwork_Skills', 0), avg_stats.get('GPA', 0)
    ]

    # 3. Generate Summary tailored to this specific switch
    fake_top_careers = [{'role': target_career, 'match': match_score}, {'role': 'N/A', 'match': 0}, {'role': 'N/A', 'match': 0}]
    ai_summary = generate_ai_summary(user_inputs, fake_top_careers, user_inputs.get('Field', 'General'))

    # 4. Generate Simulation Data (Seeded by Career Name for consistency)
    random.seed(target_career)
    years = [str(y) for y in range(2020, 2030)]
    hiring_trend = []
    base_hiring = random.randint(500, 2000)
    for _ in years:
        base_hiring += random.randint(-50, 200)
        hiring_trend.append(base_hiring)
    
    # 5. Salary Data
    sal_info = salary_stats.get(target_career, {'min': 10, 'mean': 25, 'max': 50})
    salary_trend_data = [
        sal_info['min'], (sal_info['min'] + sal_info['mean'])/2, 
        sal_info['mean'], (sal_info['mean'] + sal_info['max'])/2, sal_info['max']
    ]
    scope_score = min(int((sal_info['mean'] / 80) * 100) + random.randint(0, 20), 100)

    return jsonify({
        'role': target_career,
        'ai_summary': ai_summary,
        'user_stats': user_stats_chart,
        'career_stats': target_stats_chart,
        'years': years,
        'hiring_trend': hiring_trend,
        'salary_labels': ["Entry Level", "Junior", "Mid Level", "Senior", "Lead/Principal"],
        'salary_data': salary_trend_data,
        'scope_score': scope_score
    })

if __name__ == '__main__':
    initialize_system()
    print("Server running at http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)