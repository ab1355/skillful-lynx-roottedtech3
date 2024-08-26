import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

class AdvancedRiskPredictor:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'NeuralNetwork': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(score_func=f_regression, k=10)
        self.best_model = None

    def load_real_data(self, filepath):
        # In a real scenario, this function would load data from a CSV or database
        # For demonstration, we'll create a more realistic synthetic dataset
        np.random.seed(42)
        n_samples = 1000
        data = pd.DataFrame({
            'team_size': np.random.randint(5, 50, n_samples),
            'project_duration': np.random.randint(30, 365, n_samples),
            'project_complexity': np.random.uniform(1, 10, n_samples),
            'timeline_pressure': np.random.uniform(1, 10, n_samples),
            'budget_constraints': np.random.uniform(1, 10, n_samples),
            'team_experience': np.random.uniform(1, 10, n_samples),
            'technology_familiarity': np.random.uniform(1, 10, n_samples),
            'external_dependencies': np.random.randint(0, 5, n_samples),
            'stakeholder_involvement': np.random.uniform(1, 10, n_samples),
            'requirements_clarity': np.random.uniform(1, 10, n_samples),
            'past_project_success_rate': np.random.uniform(0, 1, n_samples),
            'risk_score': np.random.uniform(0, 100, n_samples)
        })
        return data

    def engineer_features(self, data):
        # Create new features
        data['complexity_experience_ratio'] = data['project_complexity'] / data['team_experience']
        data['pressure_duration_ratio'] = data['timeline_pressure'] / data['project_duration']
        data['budget_per_team_member'] = data['budget_constraints'] / data['team_size']
        data['technology_gap'] = 10 - data['technology_familiarity']
        data['stakeholder_req_clarity'] = data['stakeholder_involvement'] * data['requirements_clarity']
        
        return data

    def preprocess_data(self, data):
        data = self.engineer_features(data)
        
        features = [col for col in data.columns if col != 'risk_score']
        X = data[features]
        y = data['risk_score']

        X_scaled = self.scaler.fit_transform(X)
        X_selected = self.feature_selector.fit_transform(X_scaled, y)
        
        return X_selected, y

    def train_and_evaluate(self, data):
        X, y = self.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            
            results[name] = {'MSE': mse, 'R2': r2, 'CV_MSE': -cv_scores.mean()}

        self.best_model = min(results, key=lambda x: results[x]['CV_MSE'])
        
        print("Model Comparison Results:")
        for name, metrics in results.items():
            print(f"{name}:")
            print(f"  MSE: {metrics['MSE']:.4f}")
            print(f"  R2: {metrics['R2']:.4f}")
            print(f"  Cross-Validation MSE: {metrics['CV_MSE']:.4f}")
        
        print(f"\nBest Model: {self.best_model}")

    def predict_risk(self, project_data):
        project_data = self.engineer_features(project_data)
        X = self.scaler.transform(project_data)
        X_selected = self.feature_selector.transform(X)
        return self.models[self.best_model].predict(X_selected)

# Example usage
if __name__ == "__main__":
    risk_predictor = AdvancedRiskPredictor()
    
    # Load and preprocess data
    data = risk_predictor.load_real_data("dummy_path")  # In real scenario, provide actual file path
    
    # Train and evaluate models
    risk_predictor.train_and_evaluate(data)

    # Example prediction
    new_project = pd.DataFrame({
        'team_size': [20],
        'project_duration': [180],
        'project_complexity': [7.5],
        'timeline_pressure': [8],
        'budget_constraints': [6],
        'team_experience': [7],
        'technology_familiarity': [6.5],
        'external_dependencies': [2],
        'stakeholder_involvement': [8],
        'requirements_clarity': [7],
        'past_project_success_rate': [0.8]
    })

    predicted_risk = risk_predictor.predict_risk(new_project)
    print(f"\nPredicted risk for the new project: {predicted_risk[0]:.2f}")
    
    print("\nMost important features:")
    feature_names = risk_predictor.engineer_features(data.drop('risk_score', axis=1)).columns
    feature_importances = risk_predictor.feature_selector.scores_
    top_features = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)[:5]
    for name, importance in top_features:
        print(f"{name}: {importance:.2f}")