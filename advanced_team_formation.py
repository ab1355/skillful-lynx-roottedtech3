import random
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from enhanced_employee_model import Employee, Base
from performance_prediction_api import predict_performance
import networkx as nx

# Initialize database connection
engine = create_engine('sqlite:///agentzero_hr.db')
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()

def calculate_compatibility(emp1, emp2):
    """Calculate compatibility score between two employees"""
    score = 0
    
    # Age compatibility (prefer diverse age groups)
    age_diff = abs(emp1.age - emp2.age)
    score += min(age_diff, 10) / 2  # Max 5 points for age diversity
    
    # Job satisfaction compatibility
    score += 5 - abs(emp1.job_satisfaction - emp2.job_satisfaction)
    
    # Work-life balance compatibility
    score += 5 - abs(emp1.work_life_balance - emp2.work_life_balance)
    
    # Relationship satisfaction compatibility
    score += 5 - abs(emp1.relationship_satisfaction - emp2.relationship_satisfaction)
    
    # Skill complementarity (simplified - just checking if they're from different departments)
    if emp1.department != emp2.department:
        score += 10
    
    return score

def form_optimal_team(project_requirements, team_size, model, scaler):
    """Form an optimal team based on project requirements and employee data"""
    all_employees = session.query(Employee).all()
    
    # Create a graph where nodes are employees and edges are compatibility scores
    G = nx.Graph()
    for emp in all_employees:
        predicted_performance = predict_performance(emp, model, scaler)
        G.add_node(emp.id, employee=emp, performance=predicted_performance)
    
    # Add edges (compatibility scores) between all pairs of employees
    for emp1 in all_employees:
        for emp2 in all_employees:
            if emp1.id < emp2.id:  # Avoid duplicate edges
                compatibility = calculate_compatibility(emp1, emp2)
                G.add_edge(emp1.id, emp2.id, weight=compatibility)
    
    # Use a graph algorithm to find a subgraph with high compatibility and performance
    # This is a simplified approach and can be further optimized
    team = []
    remaining_employees = list(G.nodes())
    
    while len(team) < team_size and remaining_employees:
        # Select the employee with the highest predicted performance
        best_employee = max(remaining_employees, key=lambda x: G.nodes[x]['performance'])
        team.append(best_employee)
        remaining_employees.remove(best_employee)
        
        # Update remaining employees based on compatibility with the selected employee
        remaining_employees.sort(key=lambda x: G[x][best_employee]['weight'], reverse=True)
    
    # Convert employee IDs back to Employee objects
    final_team = [session.query(Employee).get(emp_id) for emp_id in team]
    
    return final_team

def print_team_info(team):
    """Print information about the formed team"""
    print("Optimal Team Composition:")
    for i, employee in enumerate(team, 1):
        print(f"Member {i}:")
        print(f"  ID: {employee.id}")
        print(f"  Age: {employee.age}")
        print(f"  Department: {employee.department}")
        print(f"  Job Role: {employee.job_role}")
        print(f"  Performance Score: {employee.performance_score}")
        print(f"  Job Satisfaction: {employee.job_satisfaction}")
        print(f"  Years at Company: {employee.years_at_company}")
        print("--------------------")

if __name__ == "__main__":
    # For testing purposes
    from performance_prediction_api import train_performance_model
    model, scaler = train_performance_model()
    
    # Example project requirements (simplified for this example)
    project_requirements = {
        "team_size": 5,
        "required_skills": ["Python", "Data Analysis", "Project Management"]
    }
    
    optimal_team = form_optimal_team(project_requirements, project_requirements["team_size"], model, scaler)
    print_team_info(optimal_team)