import json
import logging
from flask import Flask, request, jsonify
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
    return "Welcome to AgentZero HR Analytics"

@app.route('/dashboard')
def dashboard():
    dashboard_data = {
        "total_employees": len(hr_data),
        "avg_performance": calculate_avg_performance(hr_data),
        "dept_distribution": get_dept_distribution(hr_data),
        "age_distribution": get_age_distribution(hr_data),
        "tenure_distribution": get_tenure_distribution(hr_data)
    }
    return jsonify(dashboard_data)

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict_performance', methods=['POST'])
def predict_performance_route():
    try:
        data = request.json
        predicted_performance = predict_performance(data)
        return jsonify({'predicted_performance': predicted_performance})
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Error in predict_performance: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/form_team', methods=['POST'])
def form_team_route():
    try:
        data = request.json
        if 'team_size' not in data or 'required_skills' not in data:
            raise ValueError('Missing required fields: team_size and required_skills')
        optimal_team = form_optimal_team(data["team_size"], data["required_skills"])
        return jsonify({'optimal_team': optimal_team})
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Error in form_team: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT)