import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy import stats

def generate_data():
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = (X[:, 0] + X[:, 1]**2 + np.random.randn(1000) > 0).astype(int)
    return X, y

def statistical_test(X, y):
    print("1. Statistical Test:")
    t_stat, p_value = stats.ttest_ind(X[y==0][:, 0], X[y==1][:, 0])
    print(f"T-statistic: {t_stat}, p-value: {p_value}")
    
def machine_learning_test(X, y):
    print("\n2. Machine Learning Test:")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def main():
    X, y = generate_data()
    statistical_test(X, y)
    machine_learning_test(X, y)

if __name__ == "__main__":
    main()