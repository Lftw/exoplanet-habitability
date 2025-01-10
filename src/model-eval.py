from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
import pandas as pd

def evaluate_model(model_path, X_test, y_test):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    print(classification_report(y_test, y_pred))
    
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv('../data/processed/exoplanet_data.csv')
    X_test = data.drop(columns=['habitable'])
    y_test = data['habitable']
    evaluate_model('../src/model.pkl', X_test, y_test)
