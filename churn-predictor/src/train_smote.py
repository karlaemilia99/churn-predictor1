import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Cargar datos
X_train = pd.read_csv('data/X_train.csv')
X_test  = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test  = pd.read_csv('data/y_test.csv').squeeze()

# Aplicar SMOTE al train
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("=== DISTRIBUCIÓN DESPUÉS DE SMOTE ===")
print(pd.Series(y_train_bal).value_counts())

# Modelos
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':             XGBClassifier(eval_metric='logloss', random_state=42)
}

results = {}
print("\n=== ENTRENANDO CON SMOTE ===\n")

for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    results[name] = {'model': model, 'auc': auc, 'y_pred': y_pred}
    print(f"--- {name} ---")
    print(f"AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Mejor modelo
best_name = max(results, key=lambda x: results[x]['auc'])
best_model = results[best_name]['model']
print(f"\n✅ Mejor modelo con SMOTE: {best_name} (AUC: {results[best_name]['auc']:.4f})")

with open('models/best_model_smote.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Matriz de confusión
cm = confusion_matrix(y_test, results[best_name]['y_pred'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title(f'Matriz de Confusión con SMOTE — {best_name}')
plt.ylabel('Real')
plt.xlabel('Predicho')
plt.tight_layout()
plt.savefig('reports/confusion_matrix_smote.png', dpi=150)
plt.show()