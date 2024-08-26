import json
import logging
from flask import Flask, request, jsonify, render_template
from hr_data import load_hr_data, calculate_avg_performance, get_dept_distribution, get_age_distribution, get_tenure_distribution
import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format='%(asctime)s - %(levelname)s - %(message)s')

# Load HR data
hr_data = load_hr_data(config.DATA_FILE)

app = Flask(__name__)

def predict_performance(employee_data):
    if 'performance_score' not in employee_data or 'years_at_company' not in employee_data:
        raise ValueError("Missing required fields: performance_score and years_at_company")
    
    base_score = float(employee_data["performance_score"])
    years = int(employee_data["years_at_company"])
    
    if not (config.MIN_PERFORMANCE_SCORE <= base_score <= config.MAX_PERFORMANCE_SCORE):
        raise ValueError(f"Performance score must be between {config.MIN_PERFORMANCE_SCORE} and {config.MAX_PERFORMANCE_SCORE}")
    if years < 0:
        raise ValueError("Years at company cannot be negative")
    
    years_factor = min(years * 0.01, 0.2)
    return round(min(base_score + years_factor, config.MAX_PERFORMANCE_SCORE), 2)

def form_optimal_team(team_size, required_skills):
    if not isinstance(team_size, int) or team_size <= 0 or team_size > config.MAX_TEAM_SIZE:
        raise ValueError(f"Team size must be a positive integer not exceeding {config.MAX_TEAM_SIZE}")
    if not isinstance(required_skills, list) or not required_skills:
        raise ValueError("Required skills must be a non-empty list")
    
    suitable_employees = [emp for emp in hr_data if any(skill in emp["skills"] for skill in required_skills)]
    suitable_employees.sort(key=lambda x: x["performance_score"], reverse=True)
    return [{"id": emp["id"], "name": emp["name"], "skills": emp["skills"]} for emp in suitable_employees[:team_size]]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    dashboard_data = {
        "total_employees": len(hr_data),
        "avg_performance": calculate_avg_performance(hr_data),
        "dept_distribution": get_dept_distribution(hr_data),
        "age_distribution": get_age_distribution(hr_data),
        "tenure_distribution": get_tenure_distribution(hr_data)
    }
    return render_template('dashboard.html', data=dashboard_data)

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict_performance', methods=['GET', 'POST'])
def predict_performance_route():
    if request.method == 'POST':
        try:
            data = request.form
            predicted_performance = predict_performance(data)
            return render_template('predict_performance.html', prediction=predicted_performance)
        except ValueError as ve:
            return render_template('predict_performance.html', error=str(ve))
        except Exception as e:
            logging.error(f"Error in predict_performance: {str(e)}")
            return render_template('predict_performance.html', error="Internal Server Error")
    return render_template('predict_performance.html')

@app.route('/form_team', methods=['GET', 'POST'])
def form_team_route():
    if request.method == 'POST':
        try:
            data = request.form
            team_size = int(data.get('team_size', 0))
            required_skills = data.get('required_skills', '').split(',')
            optimal_team = form_optimal_team(team_size, required_skills)
            return render_template('form_team.html', team=optimal_team)
        except ValueError as ve:
            return render_template('form_team.html', error=str(ve))
        except Exception as e:
            logging.error(f"Error in form_team: {str(e)}")
            return render_template('form_team.html', error="Internal Server Error")
    return render_template('form_team.html')

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT)