import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = pd.Categorical.from_codes(y, target_names)
    
    return df, X, y, feature_names, target_names

def exploratory_data_analysis(df):
    print("Data Shape:", df.shape)
    print("\nData Info:")
    df.info()
    print("\nSummary Statistics:")
    print(df.describe())
    
    plt.figure(figsize=(12, 8))
    sns.pairplot(df, hue='species')
    plt.savefig('iris_pairplot.png')
    plt.close()
    
    print("\nPairplot saved as 'iris_pairplot.png'")

def train_and_evaluate_model(X, y, feature_names, target_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    y_pred = rf_model.predict(X_test_scaled)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': rf_model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)

def main():
    df, X, y, feature_names, target_names = load_and_prepare_data()
    exploratory_data_analysis(df)
    train_and_evaluate_model(X, y, feature_names, target_names)

if __name__ == "__main__":
    main()