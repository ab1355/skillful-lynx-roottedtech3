from flask import Flask, request, jsonify
from enhanced_employee_model import Employee, session, train_performance_model, predict_performance
from advanced_team_formation import form_optimal_team, print_team_info

app = Flask(__name__)

# Train the model on startup
model, scaler = train_performance_model()

@app.route('/predict_performance', methods=['POST'])
def predict_employee_performance():
    data = request.json
    employee = Employee(
        age=data['age'],
        years_at_company=data['years_at_company'],
        years_in_current_role=data['years_in_current_role'],
        job_satisfaction=data['job_satisfaction'],
        job_involvement=data['job_involvement'],
        relationship_satisfaction=data['relationship_satisfaction'],
        work_life_balance=data['work_life_balance']
    )
    predicted_performance = predict_performance(employee, model, scaler)
    return jsonify({'predicted_performance': predicted_performance})

@app.route('/form_team', methods=['POST'])
def form_team():
    data = request.json
    project_requirements = {
        "team_size": data['team_size'],
        "required_skills": data['required_skills']
    }
    optimal_team = form_optimal_team(project_requirements, project_requirements["team_size"], model, scaler)
    
    team_info = []
    for employee in optimal_team:
        team_info.append({
            "id": employee.id,
            "age": employee.age,
            "department": employee.department,
            "job_role": employee.job_role,
            "performance_score": employee.performance_score,
            "job_satisfaction": employee.job_satisfaction,
            "years_at_company": employee.years_at_company
        })
    
    return jsonify({'optimal_team': team_info})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)