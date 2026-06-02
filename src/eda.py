import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Vista general
print("=== SHAPE ===")
print(df.shape)

print("\n=== PRIMERAS FILAS ===")
print(df.head())

print("\n=== TIPOS DE DATOS ===")
print(df.dtypes)

print("\n=== VALORES NULOS ===")
print(df.isnull().sum())

print("\n=== DISTRIBUCIÓN DE CHURN ===")
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True).round(2))

# Configuración visual
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Exploración de Churn', fontsize=16, fontweight='bold')

# 1. Distribución de churn
df['Churn'].value_counts().plot(
    kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'], edgecolor='black'
)
axes[0].set_title('Distribución de Churn')
axes[0].set_xlabel('')
axes[0].set_ylabel('Clientes')
axes[0].tick_params(axis='x', rotation=0)

# 2. Churn por tipo de contrato
contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
contract_churn.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'], edgecolor='black')
axes[1].set_title('Churn por Tipo de Contrato')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=15)
axes[1].legend(['No Churn', 'Churn'])

# 3. Tenure vs Churn
df.boxplot(column='tenure', by='Churn', ax=axes[2], 
           boxprops=dict(color='#2c3e50'),
           medianprops=dict(color='#e74c3c', linewidth=2))
axes[2].set_title('Antigüedad vs Churn')
axes[2].set_xlabel('Churn')
axes[2].set_ylabel('Meses como cliente')
plt.suptitle('')

plt.tight_layout()
plt.savefig('reports/churn_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gráfica guardada en reports/churn_eda.png")