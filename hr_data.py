import json
import random

# Generate more realistic dummy data
def generate_hr_data(num_employees=50):
    departments = ["Sales", "Marketing", "Engineering", "HR", "Finance"]
    skills = ["Python", "Data Analysis", "Machine Learning", "Project Management", "Communication", "Leadership"]
    
    hr_data = []
    for i in range(1, num_employees + 1):
        employee = {
            "id": i,
            "name": f"Employee {i}",
            "performance_score": round(random.uniform(0.5, 1.0), 2),
            "department": random.choice(departments),
            "age": random.randint(22, 60),
            "years_at_company": random.randint(0, 20),
            "skills": random.sample(skills, random.randint(1, len(skills)))
        }
        hr_data.append(employee)
    return hr_data

# Save data to a file
def save_hr_data(data, filename='hr_data.json'):
    with open(filename, 'w') as f:
        json.dump(data, f)

# Load data from a file
def load_hr_data(filename='hr_data.json'):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found. Generating new data.")
        data = generate_hr_data()
        save_hr_data(data)
        return data

# Data processing functions
def calculate_avg_performance(data):
    return sum(emp["performance_score"] for emp in data) / len(data)

def get_dept_distribution(data):
    dept_dist = {}
    for emp in data:
        dept = emp["department"]
        dept_dist[dept] = dept_dist.get(dept, 0) + 1
    return dept_dist

def get_age_distribution(data):
    age_dist = {}
    for emp in data:
        age = emp["age"] // 10 * 10  # Group by decade
        age_dist[f"{age}-{age+9}"] = age_dist.get(f"{age}-{age+9}", 0) + 1
    return age_dist

def get_tenure_distribution(data):
    tenure_dist = {}
    for emp in data:
        years = emp["years_at_company"]
        if years < 2:
            tenure_dist["0-1 years"] = tenure_dist.get("0-1 years", 0) + 1
        elif years < 5:
            tenure_dist["2-4 years"] = tenure_dist.get("2-4 years", 0) + 1
        elif years < 10:
            tenure_dist["5-9 years"] = tenure_dist.get("5-9 years", 0) + 1
        else:
            tenure_dist["10+ years"] = tenure_dist.get("10+ years", 0) + 1
    return tenure_dist

if __name__ == "__main__":
    # Generate and save data when this script is run directly
    data = generate_hr_data()
    save_hr_data(data)
    print(f"Generated and saved data for {len(data)} employees.")