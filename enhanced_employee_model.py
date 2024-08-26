from sqlalchemy import Column, Integer, String, Float, Boolean, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

Base = declarative_base()

class Employee(Base):
    __tablename__ = 'employees'

    id = Column(Integer, primary_key=True)
    age = Column(Integer)
    gender = Column(String)
    marital_status = Column(String)
    education_level = Column(String)
    department = Column(String)
    job_role = Column(String)
    job_satisfaction = Column(Integer)
    years_at_company = Column(Integer)
    years_in_current_role = Column(Integer)
    performance_score = Column(Float)
    job_involvement = Column(Integer)
    relationship_satisfaction = Column(Integer)
    work_life_balance = Column(Integer)
    salary = Column(Float)

# Initialize database
engine = create_engine('sqlite:///agentzero_hr.db')
Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

def load_sample_data(file_path):
    # Load data from CSV (assuming you have downloaded the dataset)
    df = pd.read_csv(file_path)
    
    for _, row in df.iterrows():
        employee = Employee(
            age=row['Age'],
            gender=row['Gender'],
            marital_status=row['MaritalStatus'],
            education_level=row['Education'],
            department=row['Department'],
            job_role=row['JobRole'],
            job_satisfaction=row['JobSatisfaction'],
            years_at_company=row['YearsAtCompany'],
            years_in_current_role=row['YearsInCurrentRole'],
            performance_score=row['PerformanceRating'],
            job_involvement=row['JobInvolvement'],
            relationship_satisfaction=row['RelationshipSatisfaction'],
            work_life_balance=row['WorkLifeBalance'],
            salary=row['MonthlyIncome']
        )
        session.add(employee)
    
    session.commit()

def train_performance_model():
    # Fetch all employees
    employees = session.query(Employee).all()
    
    # Prepare data for model
    X = [[e.age, e.years_at_company, e.years_in_current_role, e.job_satisfaction, 
          e.job_involvement, e.relationship_satisfaction, e.work_life_balance] for e in employees]
    y = [e.performance_score for e in employees]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    score = model.score(X_test_scaled, y_test)
    print(f"Model R-squared score: {score}")
    
    return model, scaler

def predict_performance(employee, model, scaler):
    features = [[employee.age, employee.years_at_company, employee.years_in_current_role, 
                 employee.job_satisfaction, employee.job_involvement, 
                 employee.relationship_satisfaction, employee.work_life_balance]]
    features_scaled = scaler.transform(features)
    return model.predict(features_scaled)[0]

if __name__ == "__main__":
    # Uncomment the following line to load sample data (make sure you have the CSV file)
    # load_sample_data('path_to_your_hr_dataset.csv')
    
    model, scaler = train_performance_model()
    
    # Example: Predict performance for a new employee
    new_employee = Employee(
        age=30,
        years_at_company=5,
        years_in_current_role=3,
        job_satisfaction=4,
        job_involvement=4,
        relationship_satisfaction=3,
        work_life_balance=3
    )
    
    predicted_performance = predict_performance(new_employee, model, scaler)
    print(f"Predicted performance score: {predicted_performance}")