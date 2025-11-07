import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    recall_score, 
    roc_auc_score, 
    roc_curve,
    classification_report
)
import warnings
import os

warnings.filterwarnings('ignore')
output_dir = 'resultados_imagens'
os.makedirs(output_dir, exist_ok=True)
print(f"Pasta '{output_dir}' criada (ou já existe).")

data = fetch_california_housing(as_frame=True)
df = data.frame

mediana = df['MedHouseVal'].median()
df['PriceCategory'] = (df['MedHouseVal'] > mediana).astype(int)

X = df.drop(['MedHouseVal', 'PriceCategory'], axis=1)
y = df['PriceCategory']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model_lr = LogisticRegression(random_state=42, max_iter=1000)
model_rf = RandomForestClassifier(random_state=42)
model_xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

print("Treinando modelos...")
model_lr.fit(X_train_scaled, y_train)
model_rf.fit(X_train_scaled, y_train)
model_xgb.fit(X_train_scaled, y_train)
print("Treinamento concluído.")
print("\nGerando avaliações...")

results = {}

def evaluate_model(model, name, X_train, y_train, X_val, y_val):
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_val = model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'Accuracy (Treino)': accuracy_score(y_train, y_pred_train),
        'Accuracy (Val)': accuracy_score(y_val, y_pred_val),
        'F1 (Treino)': f1_score(y_train, y_pred_train),
        'F1 (Val)': f1_score(y_val, y_pred_val),
        'Recall (Treino)': recall_score(y_train, y_pred_train),
        'Recall (Val)': recall_score(y_val, y_pred_val),
        'AUC (Treino)': roc_auc_score(y_train, y_proba_train),
        'AUC (Val)': roc_auc_score(y_val, y_proba_val),
    }
    results[name] = metrics
    return metrics

evaluate_model(model_lr, 'Logistic Regression', X_train_scaled, y_train, X_val_scaled, y_val)
evaluate_model(model_rf, 'Random Forest', X_train_scaled, y_train, X_val_scaled, y_val)
evaluate_model(model_xgb, 'XGBoost', X_train_scaled, y_train, X_val_scaled, y_val)

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

cv_lr = cross_val_score(model_lr, X_train_scaled, y_train, cv=kf, scoring='roc_auc')
cv_rf = cross_val_score(model_rf, X_train_scaled, y_train, cv=kf, scoring='roc_auc')
cv_xgb = cross_val_score(model_xgb, X_train_scaled, y_train, cv=kf, scoring='roc_auc')

results['Logistic Regression']['CV Média (AUC)'] = cv_lr.mean()
results['Logistic Regression']['CV Std (AUC)'] = cv_lr.std()
results['Random Forest']['CV Média (AUC)'] = cv_rf.mean()
results['Random Forest']['CV Std (AUC)'] = cv_rf.std()
results['XGBoost']['CV Média (AUC)'] = cv_xgb.mean()
results['XGBoost']['CV Std (AUC)'] = cv_xgb.std()

y_proba_lr = model_lr.predict_proba(X_val_scaled)[:, 1]
y_proba_rf = model_rf.predict_proba(X_val_scaled)[:, 1]
y_proba_xgb = model_xgb.predict_proba(X_val_scaled)[:, 1]

fpr_lr, tpr_lr, _ = roc_curve(y_val, y_proba_lr)
auc_lr = roc_auc_score(y_val, y_proba_lr)

fpr_rf, tpr_rf, _ = roc_curve(y_val, y_proba_rf)
auc_rf = roc_auc_score(y_val, y_proba_rf)

fpr_xgb, tpr_xgb, _ = roc_curve(y_val, y_proba_xgb)
auc_xgb = roc_auc_score(y_val, y_proba_xgb)

plt.figure(figsize=(10, 7))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.4f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.4f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.50)')
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC (Conjunto de Validação)')
plt.legend()
plt.grid()

roc_curve_path = os.path.join(output_dir, 'curva_roc.png')
plt.savefig(roc_curve_path)
print(f"\nGráfico da Curva ROC salvo em: {roc_curve_path}")
plt.close() 

noise_level = 0.2
X_val_noisy = X_val_scaled + np.random.normal(0, noise_level, X_val_scaled.shape)

auc_noisy_lr = roc_auc_score(y_val, model_lr.predict_proba(X_val_noisy)[:, 1])
auc_noisy_rf = roc_auc_score(y_val, model_rf.predict_proba(X_val_noisy)[:, 1])
auc_noisy_xgb = roc_auc_score(y_val, model_xgb.predict_proba(X_val_noisy)[:, 1])

results['Logistic Regression']['AUC (Generalização)'] = auc_noisy_lr
results['Random Forest']['AUC (Generalização)'] = auc_noisy_rf
results['XGBoost']['AUC (Generalização)'] = auc_noisy_xgb

def evaluate_test_set(model, name):
    y_pred_test = model.predict(X_test_scaled)
    y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
    
    results[name]['Accuracy (TESTE)'] = accuracy_score(y_test, y_pred_test)
    results[name]['F1 (TESTE)'] = f1_score(y_test, y_pred_test)
    results[name]['Recall (TESTE)'] = recall_score(y_test, y_pred_test)
    results[name]['AUC (TESTE)'] = roc_auc_score(y_test, y_proba_test)
    
    print(f"\n--- Relatório de Classificação (TESTE): {name} ---")
    print(classification_report(y_test, y_pred_test))

evaluate_test_set(model_lr, 'Logistic Regression')
evaluate_test_set(model_rf, 'Random Forest')
evaluate_test_set(model_xgb, 'XGBoost')

final_table = pd.DataFrame(results).T

columns_order = [
    'AUC (TESTE)',
    'Accuracy (TESTE)',
    'F1 (TESTE)',
    'AUC (Treino)',
    'AUC (Val)',
    'CV Média (AUC)',
    'CV Std (AUC)',
    'AUC (Generalização)'
]
final_table = final_table[columns_order]

fig, ax = plt.subplots(figsize=(14, 4)) 
ax.axis('tight')
ax.axis('off')

final_table_formatted = final_table.round(4)

the_table = ax.table(
    cellText=final_table_formatted.values, 
    colLabels=final_table_formatted.columns, 
    rowLabels=final_table_formatted.index, 
    cellLoc='center', 
    loc='center'
)

the_table.auto_set_font_size(False)
the_table.set_fontsize(9)
the_table.scale(1.1, 1.1)

plt.title('Tabela Comparativa Final', y=1.08, fontsize=14)

table_image_path = os.path.join(output_dir, 'tabela_comparativa.png')
plt.savefig(table_image_path, bbox_inches='tight', dpi=200)
print(f"Tabela Comparativa salva em: {table_image_path}")
plt.close()

print("\n\n--- TABELA COMPARATIVA FINAL (Markdown) ---")
print(final_table.to_markdown(floatfmt=".4f"))
