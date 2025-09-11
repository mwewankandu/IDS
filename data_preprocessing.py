# data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(filepath="creditcard.csv"):
    """
    Load and preprocess the fraud dataset.
    Returns: X_train_res, X_test, y_train_res, y_test
    """
    # Load data
    df = pd.read_csv(filepath)

    # Separate features (X) and target (y)
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle class imbalance with SMOTE (only on training data!)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test

if __name__ == "__main__":
    # Test the function
    X_train_res, X_test, y_train_res, y_test = load_and_preprocess_data()
    print("Training data shape:", X_train_res.shape)
    print("Test data shape:", X_test.shape)