import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def load_and_clean_data():
    """
    Load and clean the breast cancer dataset.
    Standardize the feature data.
    
    Returns:
        X_scaled (ndarray): Standardized feature data.
        y (ndarray): Target labels.
    """
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Check if data is loaded correctly
    if X is not None and y is not None:
        print("Data loaded successfully.")
    else:
        print("Error loading data.")
        return None, None

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data cleaned and standardized.")
    
    return X_scaled, y

def perform_pca(X, n_components=2):
    """
    Perform PCA to reduce the dataset to the specified number of components.
    
    Args:
        X (ndarray): Standardized feature data.
        n_components (int): Number of PCA components.
    
    Returns:
        X_pca (ndarray): Transformed feature data with reduced dimensions.
    """
    try:
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        if X_pca.shape[1] == n_components:
            print(f"PCA reduction to {n_components} components was successful.")
        else:
            print(f"PCA reduction did not result in {n_components} components.")
        return X_pca
    except Exception as e:
        print(f"Error in PCA reduction: {e}")
        return None

def split_data(X, y, test_size=0.3, random_state=42):
    """
    Split the dataset into training and testing sets.
    
    Args:
        X (ndarray): Feature data.
        y (ndarray): Target labels.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        X_train, X_test, y_train, y_test: Split datasets.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print("Data split into training and testing sets successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error splitting data: {e}")
        return None, None, None, None

def train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a logistic regression model.
    
    Args:
        X_train (ndarray): Training feature data.
        X_test (ndarray): Testing feature data.
        y_train (ndarray): Training target labels.
        y_test (ndarray): Testing target labels.
    """
    try:
        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)
        y_pred = log_reg.predict(X_test)
        print("Logistic regression model trained successfully.")
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > 0.95:
            print(f"Model accuracy is high: {accuracy}")
        else:
            print(f"Model accuracy is low: {accuracy}")
    except Exception as e:
        print(f"Error training logistic regression model: {e}")

def main():
    """
    Main function to execute the script interactively.
    """
    while True:
        print("\nSelect an option:")
        print("1. Load and clean data")
        print("2. Perform PCA")
        print("3. Split data")
        print("4. Train and evaluate logistic regression")
        print("5. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            global X, y
            X, y = load_and_clean_data()
        elif choice == '2':
            if 'X' in globals():
                n_components = int(input("Enter the number of PCA components: "))
                global X_pca
                X_pca = perform_pca(X, n_components)
            else:
                print("Data not loaded. Please load and clean data first.")
        elif choice == '3':
            if 'X_pca' in globals():
                global X_train, X_test, y_train, y_test
                X_train, X_test, y_train, y_test = split_data(X_pca, y)
            else:
                print("PCA not performed. Please perform PCA first.")
        elif choice == '4':
            if 'X_train' in globals() and 'y_train' in globals():
                train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test)
            else:
                print("Data not split. Please split data first.")
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
