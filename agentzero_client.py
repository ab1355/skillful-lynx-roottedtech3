import requests
import json

API_BASE_URL = "http://0.0.0.0:8080"

def predict_performance(employee_data):
    response = requests.post(f"{API_BASE_URL}/predict_performance", json=employee_data)
    if response.status_code == 200:
        return response.json()['predicted_performance']
    else:
        print(f"Error: {response.status_code}")
        return None

def form_team(project_requirements):
    response = requests.post(f"{API_BASE_URL}/form_team", json=project_requirements)
    if response.status_code == 200:
        return response.json()['optimal_team']
    else:
        print(f"Error: {response.status_code}")
        return None

def print_team_info(team):
    print("Optimal Team Composition:")
    for i, member in enumerate(team, 1):
        print(f"Member {i}:")
        print(json.dumps(member, indent=2))
        print("--------------------")

if __name__ == "__main__":
    # Example: Predict performance for an employee
    employee_data = {
        "age": 30,
        "years_at_company": 5,
        "years_in_current_role": 3,
        "job_satisfaction": 4,
        "job_involvement": 4,
        "relationship_satisfaction": 3,
        "work_life_balance": 3
    }
    predicted_performance = predict_performance(employee_data)
    print(f"Predicted performance: {predicted_performance}")

    # Example: Form an optimal team
    project_requirements = {
        "team_size": 5,
        "required_skills": ["Python", "Data Analysis", "Project Management"]
    }
    optimal_team = form_team(project_requirements)
    if optimal_team:
        print_team_info(optimal_team)