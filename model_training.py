import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
def load_data():
    try:
        # Load dataset (replace with your actual data source)
        df = pd.read_csv('diabetes.csv')
        print("Dataset loaded successfully")
        print(f"Shape: {df.shape}")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

# Preprocess the data
def preprocess_data(df):
    # Replace zeros with median for specific columns
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_columns:
        df[col] = df[col].replace(0, np.nan)
        median = df[col].median()
        df[col] = df[col].fillna(median)
    
    # Feature scaling
    scaler = StandardScaler()
    features = df.drop('Outcome', axis=1)
    scaled_features = scaler.fit_transform(features)
    
    # Create new DataFrame with scaled features
    df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
    df_scaled['Outcome'] = df['Outcome']
    
    return df_scaled

# Train the model
def train_model(X_train, y_train):
    # Initialize Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    # Train the model
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Save the model
def save_model(model, filename='diabetes_model.pkl'):
    try:
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved successfully as {filename}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

# Main function
def main():
    # Step 1: Load data
    df = load_data()
    if df is None:
        return
    
    # Step 2: Preprocess data
    df_processed = preprocess_data(df)
    
    # Split data into features and target
    X = df_processed.drop('Outcome', axis=1)
    y = df_processed['Outcome']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Step 3: Train model
    print("\nTraining model...")
    model = train_model(X_train, y_train)
    
    # Step 4: Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # Step 5: Save model
    save_model(model)

if __name__ == "__main__":
    main()