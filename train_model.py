import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from utils import ChurnPreprocessor
import joblib

def train_and_save():
    # Load data
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Preprocess
    preprocessor = ChurnPreprocessor()
    X, y = preprocessor.fit(df)
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )
    
    # Train models
    models = {
        'logistic': LogisticRegression(max_iter=1000),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'xgb': xgb.XGBClassifier(n_estimators=150, max_depth=6, random_state=42)
    }
    
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"{name}: {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_model = model
    
    # Save best model + preprocessor
    joblib.dump(best_model, 'models/churn_model_xgb.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    
    print(f"\nBest model: XGBoost (Accuracy: {best_score:.3f})")
    print("Saved to models/ folder")
    
if __name__ == "__main__":
    train_and_save()
