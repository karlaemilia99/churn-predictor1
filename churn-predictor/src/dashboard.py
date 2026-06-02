import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# Cargar datos
X_train = pd.read_csv('data/X_train.csv')
X_test  = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test  = pd.read_csv('data/y_test.csv').squeeze()

# SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Entrenar modelos
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':             XGBClassifier(eval_metric='logloss', random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    results[name] = {'model': model, 'auc': auc, 'fpr': fpr, 'tpr': tpr}

# ── DASHBOARD ──────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0f1117')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

COLORS = ['#00d4ff', '#ff6b6b', '#51cf66']
TEXT   = '#e0e0e0'
GRID   = '#2a2a3e'

def style_ax(ax):
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, linestyle='--', linewidth=0.5)

# 1. Curvas ROC
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1)
for (name, res), color in zip(results.items(), COLORS):
    ax1.plot(res['fpr'], res['tpr'], color=color, lw=2,
             label=f"{name} ({res['auc']:.3f})")
ax1.plot([0,1],[0,1], 'w--', lw=1, alpha=0.4)
ax1.set_title('Curvas ROC', fontweight='bold')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(fontsize=8, labelcolor=TEXT, facecolor='#1a1a2e', edgecolor=GRID)

# 2. Comparación AUC
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2)
names = list(results.keys())
aucs  = [results[n]['auc'] for n in names]
bars  = ax2.barh(names, aucs, color=COLORS, edgecolor='none', height=0.5)
ax2.set_xlim(0.75, 0.86)
ax2.set_title('Comparación AUC-ROC', fontweight='bold')
ax2.set_xlabel('AUC-ROC')
for bar, val in zip(bars, aucs):
    ax2.text(val + 0.001, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', color=TEXT, fontsize=9)

# 3. Feature Importance (Random Forest)
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3)
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
top10 = importances.nlargest(10).sort_values()
ax3.barh(top10.index, top10.values, color='#00d4ff', edgecolor='none', height=0.6)
ax3.set_title('Top 10 Features — Random Forest', fontweight='bold')
ax3.set_xlabel('Importancia')

# 4. Distribución Churn por Tenure
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4)
df_orig = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df_orig['TotalCharges'] = pd.to_numeric(df_orig['TotalCharges'], errors='coerce')
churn_yes = df_orig[df_orig['Churn'] == 'Yes']['tenure']
churn_no  = df_orig[df_orig['Churn'] == 'No']['tenure']
ax4.hist(churn_no,  bins=30, alpha=0.6, color='#51cf66', label='No Churn', edgecolor='none')
ax4.hist(churn_yes, bins=30, alpha=0.6, color='#ff6b6b', label='Churn',    edgecolor='none')
ax4.set_title('Distribución Tenure vs Churn', fontweight='bold')
ax4.set_xlabel('Meses como cliente')
ax4.set_ylabel('Clientes')
ax4.legend(fontsize=8, labelcolor=TEXT, facecolor='#1a1a2e', edgecolor=GRID)

# 5. Churn por contrato
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5)
contract_data = df_orig.groupby('Contract')['Churn'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).reset_index()
contract_data.columns = ['Contract', 'ChurnRate']
ax5.bar(contract_data['Contract'], contract_data['ChurnRate'],
        color=COLORS, edgecolor='none', width=0.5)
ax5.set_title('Tasa de Churn por Contrato (%)', fontweight='bold')
ax5.set_xlabel('')
ax5.set_ylabel('Churn %')
for i, row in contract_data.iterrows():
    ax5.text(i, row['ChurnRate'] + 0.5, f"{row['ChurnRate']:.1f}%",
             ha='center', color=TEXT, fontsize=9)

# 6. Monthly Charges vs Churn
ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6)
ax6.hist(df_orig[df_orig['Churn']=='No']['MonthlyCharges'],
         bins=30, alpha=0.6, color='#51cf66', label='No Churn', edgecolor='none')
ax6.hist(df_orig[df_orig['Churn']=='Yes']['MonthlyCharges'],
         bins=30, alpha=0.6, color='#ff6b6b', label='Churn',    edgecolor='none')
ax6.set_title('Cargo Mensual vs Churn', fontweight='bold')
ax6.set_xlabel('USD / mes')
ax6.set_ylabel('Clientes')
ax6.legend(fontsize=8, labelcolor=TEXT, facecolor='#1a1a2e', edgecolor=GRID)

# Título principal
fig.text(0.5, 0.97, 'Customer Churn Prediction — Dashboard',
         ha='center', fontsize=16, fontweight='bold', color=TEXT)
fig.text(0.5, 0.94, 'Telco Dataset  |  Logistic Regression · Random Forest · XGBoost  |  SMOTE Balancing',
         ha='center', fontsize=9, color='#888888')

plt.savefig('reports/dashboard.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Dashboard guardado en reports/dashboard.png")