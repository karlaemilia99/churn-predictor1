import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Cargar datos
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 1. TotalCharges tiene espacios vacíos — convertir a número
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# 2. Eliminar columnas que no aportan
df.drop(columns=['customerID'], inplace=True)

# 3. Convertir columnas categóricas a números
le = LabelEncoder()
categorical_cols = df.select_dtypes(include='object').columns.tolist()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 4. Separar X (features) e y (target)
X = df.drop(columns=['Churn'])
y = df['Churn']


# Rellenar cualquier NaN restante
X = X.fillna(X.median())
# 5. Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=== PREPROCESAMIENTO COMPLETO ===")
print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"\nDistribución en train:")
print(y_train.value_counts(normalize=True).round(2))

# 6. Guardar datos procesados
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

print("\nArchivos guardados en data/")