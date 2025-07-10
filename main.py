import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

class DiabetesPredictionSystem:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.model_scores = {}

    def load_data(self, path=None):
        url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
        cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        self.data = pd.read_csv(path or url, names=cols)

    def preprocess(self):
        for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            self.data[col].replace(0, np.nan, inplace=True)
            self.data[col].fillna(self.data[col].median(), inplace=True)
        X = self.data.drop('Outcome', axis=1)
        y = self.data['Outcome']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def train_models(self):
        models = {
            'LogReg': LogisticRegression(),
            'RandomForest': RandomForestClassifier(),
            'SVM': SVC(probability=True),
            'KNN': KNeighborsClassifier(),
            'DecisionTree': DecisionTreeClassifier()
        }
        for name, model in models.items():
            if name in ['LogReg', 'SVM', 'KNN']:
                model.fit(self.X_train_scaled, self.y_train)
                preds = model.predict(self.X_test_scaled)
                probs = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                model.fit(self.X_train, self.y_train)
                preds = model.predict(self.X_test)
                probs = model.predict_proba(self.X_test)[:, 1]
            acc = accuracy_score(self.y_test, preds)
            auc = roc_auc_score(self.y_test, probs)
            self.models[name] = model
            self.model_scores[name] = {'acc': acc, 'auc': auc}

    def evaluate(self):
        print("\n📊 Model Performance")
        for name, score in self.model_scores.items():
            print(f"{name}: Accuracy = {score['acc']:.4f}, AUC = {score['auc']:.4f}")
        best = max(self.model_scores, key=lambda x: self.model_scores[x]['acc'])
        print(f"\n✅ Best Model: {best}")
        y_pred = self.models[best].predict(self.X_test_scaled if best in ['LogReg', 'SVM', 'KNN'] else self.X_test)
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

    def predict(self, patient_data):
        best = max(self.model_scores, key=lambda x: self.model_scores[x]['acc'])
        model = self.models[best]
        input_df = pd.DataFrame([patient_data], columns=self.data.columns[:-1])
        input_scaled = self.scaler.transform(input_df) if best in ['LogReg', 'SVM', 'KNN'] else input_df
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        return f"Prediction: {'Diabetes' if pred else 'No Diabetes'}, Probability: {prob:.2%}, Model: {best}"

def main():
    system = DiabetesPredictionSystem()
    system.load_data()
    system.preprocess()
    system.train_models()
    system.evaluate()

    print("\n🔮 Predict Diabetes from Your Input")
    try:
        pregnancies = float(input("Enter number of pregnancies: "))
        glucose = float(input("Enter glucose level: "))
        blood_pressure = float(input("Enter blood pressure: "))
        skin_thickness = float(input("Enter skin thickness: "))
        insulin = float(input("Enter insulin level: "))
        bmi = float(input("Enter BMI: "))
        dpf = float(input("Enter diabetes pedigree function: "))
        age = float(input("Enter age: "))

        patient = [pregnancies, glucose, blood_pressure, skin_thickness,
                   insulin, bmi, dpf, age]

        result = system.predict(patient)
        print("\n🎯 Prediction Result:")
        print(result)

    except ValueError:
        print("❌ Please enter valid numeric values.")
    except Exception as e:
        print(f"⚠️ An error occurred: {e}")

if __name__ == "__main__":
    main()
