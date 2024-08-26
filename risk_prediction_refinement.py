import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

class RefinedRiskPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def preprocess_data(self, data):
        # Assuming 'data' is a pandas DataFrame with relevant features
        # Add more sophisticated feature engineering here
        features = ['team_size', 'project_complexity', 'timeline_pressure', 'budget_constraints',
                    'team_experience', 'technology_familiarity', 'external_dependencies']
        
        X = data[features]
        y = data['risk_score']

        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def train(self, data):
        X, y = self.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Model Performance - MSE: {mse:.4f}, R2: {r2:.4f}")

    def predict_risk(self, project_data):
        # Preprocess the input data
        X = self.scaler.transform(project_data)
        return self.model.predict(X)

# Example usage
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(42)
    n_samples = 1000
    data = pd.DataFrame({
        'team_size': np.random.randint(5, 50, n_samples),
        'project_complexity': np.random.uniform(1, 10, n_samples),
        'timeline_pressure': np.random.uniform(1, 10, n_samples),
        'budget_constraints': np.random.uniform(1, 10, n_samples),
        'team_experience': np.random.uniform(1, 10, n_samples),
        'technology_familiarity': np.random.uniform(1, 10, n_samples),
        'external_dependencies': np.random.randint(0, 5, n_samples),
        'risk_score': np.random.uniform(0, 100, n_samples)
    })

    risk_predictor = RefinedRiskPredictor()
    risk_predictor.train(data)

    # Example prediction
    new_project = pd.DataFrame({
        'team_size': [20],
        'project_complexity': [7.5],
        'timeline_pressure': [8],
        'budget_constraints': [6],
        'team_experience': [7],
        'technology_familiarity': [6.5],
        'external_dependencies': [2]
    })

    predicted_risk = risk_predictor.predict_risk(new_project)
    print(f"Predicted risk for the new project: {predicted_risk[0]:.2f}")