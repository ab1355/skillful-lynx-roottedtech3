import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

class AdvancedRiskPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.preprocessor = None
        self.feature_names = None

    def load_llm_prompting_data(self):
        np.random.seed(42)
        n_samples = 1000
        data = pd.DataFrame({
            'prompt_length': np.random.randint(10, 500, n_samples),
            'prompt_complexity': np.random.uniform(1, 10, n_samples),
            'target_task_difficulty': np.random.uniform(1, 10, n_samples),
            'model_size': np.random.choice(['small', 'medium', 'large'], n_samples),
            'context_relevance': np.random.uniform(0, 1, n_samples),
            'instruction_clarity': np.random.uniform(1, 10, n_samples),
            'expected_output_length': np.random.randint(50, 1000, n_samples),
            'domain_specificity': np.random.uniform(1, 10, n_samples),
            'creativity_required': np.random.uniform(1, 10, n_samples),
            'time_constraint': np.random.randint(1, 60, n_samples),
            'accuracy_score': np.random.uniform(0, 100, n_samples),
            'creativity_score': np.random.uniform(0, 100, n_samples),
            'relevance_score': np.random.uniform(0, 100, n_samples)
        })
        return data

    def engineer_features(self, data):
        data['complexity_clarity_ratio'] = data['prompt_complexity'] / data['instruction_clarity']
        data['time_pressure'] = data['expected_output_length'] / data['time_constraint']
        return data

    def preprocess_data(self, data):
        data = self.engineer_features(data)
        
        numeric_features = [col for col in data.columns if data[col].dtype in ['int64', 'float64'] and col not in ['accuracy_score', 'creativity_score', 'relevance_score']]
        categorical_features = [col for col in data.columns if data[col].dtype == 'object' or data[col].dtype.name == 'category']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        X = data.drop(['accuracy_score', 'creativity_score', 'relevance_score'], axis=1)
        y = data[['accuracy_score', 'creativity_score', 'relevance_score']]

        X_preprocessed = self.preprocessor.fit_transform(X)
        self.feature_names = self.preprocessor.get_feature_names_out()
        
        return X_preprocessed, y

    def train_and_evaluate(self, data):
        X, y = self.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        self.visualize_feature_importance()

    def predict_prompt_effectiveness(self, prompt_data):
        prompt_data = self.engineer_features(prompt_data)
        X_preprocessed = self.preprocessor.transform(prompt_data)
        
        predictions = self.model.predict(X_preprocessed)[0]
        return {
            'accuracy_score': predictions[0],
            'creativity_score': predictions[1],
            'relevance_score': predictions[2],
            'overall_effectiveness': np.mean(predictions)
        }

    def visualize_feature_importance(self):
        importances = np.abs(self.model.coef_).mean(axis=0)
        feature_importance = pd.DataFrame({'feature': self.feature_names, 'importance': importances})
        feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Top 10 Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

if __name__ == "__main__":
    risk_predictor = AdvancedRiskPredictor()
    data = risk_predictor.load_llm_prompting_data()
    risk_predictor.train_and_evaluate(data)

    new_prompt = pd.DataFrame({
        'prompt_length': [250],
        'prompt_complexity': [7],
        'target_task_difficulty': [8],
        'model_size': ['large'],
        'context_relevance': [0.9],
        'instruction_clarity': [9],
        'expected_output_length': [500],
        'domain_specificity': [6],
        'creativity_required': [8],
        'time_constraint': [30]
    })

    predicted_effectiveness = risk_predictor.predict_prompt_effectiveness(new_prompt)
    print("\nPredicted prompt effectiveness scores:")
    for key, value in predicted_effectiveness.items():
        print(f"{key}: {value:.2f}")