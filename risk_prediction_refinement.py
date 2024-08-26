import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

class AdvancedRiskPredictor:
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=5000, random_state=42),
            'XGBoost': XGBRegressor(random_state=42),
            'LightGBM': LGBMRegressor(random_state=42)
        }
        self.preprocessor = None
        self.feature_selector = SelectKBest(score_func=f_regression, k=10)
        self.best_model = None
        self.prompt_length_bins = None
        self.time_constraint_bins = None

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
            'prompt_effectiveness_score': np.random.uniform(0, 100, n_samples)
        })
        return data

    def engineer_features(self, data):
        data['complexity_clarity_ratio'] = data['prompt_complexity'] / data['instruction_clarity']
        data['time_pressure'] = data['expected_output_length'] / data['time_constraint']
        data['creativity_domain_interaction'] = data['creativity_required'] * data['domain_specificity']
        
        # Log transform for skewed numeric features
        skewed_features = ['prompt_length', 'expected_output_length', 'time_constraint']
        for feature in skewed_features:
            data[f'{feature}_log'] = np.log1p(data[feature])
        
        # Binning for numeric features (only if more than one unique value)
        if data['prompt_length'].nunique() > 1:
            data['prompt_length_bins'] = pd.qcut(data['prompt_length'], q=4, labels=['short', 'medium', 'long', 'very_long'])
            self.prompt_length_bins = data['prompt_length_bins'].cat.categories
        if data['time_constraint'].nunique() > 1:
            data['time_constraint_bins'] = pd.qcut(data['time_constraint'], q=3, labels=['short', 'medium', 'long'])
            self.time_constraint_bins = data['time_constraint_bins'].cat.categories
        
        # Interaction terms
        data['length_complexity'] = data['prompt_length'] * data['prompt_complexity']
        data['difficulty_clarity'] = data['target_task_difficulty'] * data['instruction_clarity']
        
        return data

    def preprocess_data(self, data):
        data = self.engineer_features(data)
        
        numeric_features = [col for col in data.columns if data[col].dtype in ['int64', 'float64'] and col != 'prompt_effectiveness_score']
        categorical_features = [col for col in data.columns if data[col].dtype == 'object' or data[col].dtype.name == 'category']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('power', PowerTransformer(method='yeo-johnson')),
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

        X = data.drop('prompt_effectiveness_score', axis=1)
        y = data['prompt_effectiveness_score']

        X_preprocessed = self.preprocessor.fit_transform(X)
        X_selected = self.feature_selector.fit_transform(X_preprocessed, y)
        
        return X_selected, y

    def train_and_evaluate(self, data):
        X, y = self.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            
            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'CV_MSE': -cv_scores.mean()
            }

        self.best_model = min(results, key=lambda x: results[x]['CV_MSE'])
        
        print("Model Comparison Results:")
        for name, metrics in results.items():
            print(f"{name}:")
            print(f"  MSE: {metrics['MSE']:.4f}")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  MAE: {metrics['MAE']:.4f}")
            print(f"  R2: {metrics['R2']:.4f}")
            print(f"  Cross-Validation MSE: {metrics['CV_MSE']:.4f}")
        
        print(f"\nBest Model: {self.best_model}")

        # Visualize model performance
        self.visualize_model_performance(results)

    def predict_prompt_effectiveness(self, prompt_data):
        prompt_data = self.engineer_features(prompt_data)
        
        # Add missing binned columns with a default value
        if 'prompt_length_bins' not in prompt_data.columns and self.prompt_length_bins is not None:
            prompt_data['prompt_length_bins'] = self.prompt_length_bins[0]
        if 'time_constraint_bins' not in prompt_data.columns and self.time_constraint_bins is not None:
            prompt_data['time_constraint_bins'] = self.time_constraint_bins[0]
        
        X_preprocessed = self.preprocessor.transform(prompt_data)
        X_selected = self.feature_selector.transform(X_preprocessed)
        return self.models[self.best_model].predict(X_selected)

    def visualize_feature_importance(self, data):
        X, y = self.preprocess_data(data)
        feature_names = self.preprocessor.get_feature_names_out()
        feature_importances = self.feature_selector.scores_

        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importances, y=feature_names)
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

    def visualize_correlation_matrix(self, data):
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation Matrix of Numeric Features')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        plt.close()

    def visualize_model_performance(self, results):
        model_names = list(results.keys())
        mse_scores = [results[name]['MSE'] for name in model_names]
        r2_scores = [results[name]['R2'] for name in model_names]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        sns.barplot(x=model_names, y=mse_scores, ax=ax1)
        ax1.set_title('MSE Scores by Model')
        ax1.set_ylabel('MSE')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

        sns.barplot(x=model_names, y=r2_scores, ax=ax2)
        ax2.set_title('R2 Scores by Model')
        ax2.set_ylabel('R2')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('model_performance_comparison.png')
        plt.close()

# Example usage
if __name__ == "__main__":
    risk_predictor = AdvancedRiskPredictor()
    
    # Load and preprocess data
    data = risk_predictor.load_llm_prompting_data()
    
    # Visualize correlation matrix
    risk_predictor.visualize_correlation_matrix(data)

    # Train and evaluate models
    risk_predictor.train_and_evaluate(data)

    # Visualize feature importance
    risk_predictor.visualize_feature_importance(data)

    # Example prediction
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
    print(f"\nPredicted prompt effectiveness score: {predicted_effectiveness[0]:.2f}")

    print("\nMost important features:")
    feature_names = risk_predictor.preprocessor.get_feature_names_out()
    feature_importances = risk_predictor.feature_selector.scores_
    top_features = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)[:5]
    for name, importance in top_features:
        print(f"{name}: {importance:.2f}")