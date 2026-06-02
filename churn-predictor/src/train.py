import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Cargar datos
X_train = pd.read_csv('data/X_train.csv')
X_test  = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test  = pd.read_csv('data/y_test.csv').squeeze()

# Definir modelos
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':             XGBClassifier(eval_metric='logloss', random_state=42)
}

results = {}

print("=== ENTRENANDO MODELOS ===\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    results[name] = {'model': model, 'auc': auc, 'y_pred': y_pred}
    print(f"--- {name} ---")
    print(f"AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Guardar el mejor modelo
best_name = max(results, key=lambda x: results[x]['auc'])
best_model = results[best_name]['model']
print(f"\n✅ Mejor modelo: {best_name} (AUC: {results[best_name]['auc']:.4f})")

os.makedirs('models', exist_ok=True)
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("Modelo guardado en models/best_model.pkl")

# Matriz de confusión del mejor modelo
cm = confusion_matrix(y_test, results[best_name]['y_pred'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title(f'Matriz de Confusión — {best_name}')
plt.ylabel('Real')
plt.xlabel('Predicho')
plt.tight_layout()
plt.savefig('reports/confusion_matrix.png', dpi=150)
plt.show()
print("Gráfica guardada en reports/")