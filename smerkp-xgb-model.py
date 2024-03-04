from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def load_and_evaluate_model(model_path, test_data_path):
    # Load the saved object
    loaded_object = load(model_path)
    
    # Check if the loaded object is a list and extract the model
    model = loaded_object[0] if isinstance(loaded_object, list) and len(loaded_object) > 0 else loaded_object
    
    # Validate the model
    if not hasattr(model, 'predict'):
        raise ValueError("The loaded object is not a valid model.")
    
    # Load test data
    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop(['Def'], axis=1)
    y_test = test_df['Def']
    
    
     # Measure prediction time
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    prediction_time = end_time - start_time 
    
    # Print evaluation metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Prediction Time:", prediction_time, "seconds")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Plot ROC Curve and AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

# Usage
model_path = '/smerkp_xgb_model-2.joblib'  # Confirm this is the correct path where your model is saved
test_data_path = 'csv_result-PC1-New1.csv'  # Confirm this is the correct path of your test data
load_and_evaluate_model(model_path, test_data_path)
