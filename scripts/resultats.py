import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Fonction pour charger les résultats à partir d'un DataFrame
def get_roc_data(df, label_col, prob_col):
    y_true = df[label_col]
    y_pred_proba = df[prob_col]
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    return fpr, tpr, roc_auc

# Chargement des DataFrames des résultats des modèles CNN et EfficientNetB0
df_cnn = pd.read_csv("C:\\Users\\estel\\STAGE\\Dataset\\predictions_cnn.csv")
df_efficientnet = pd.read_csv("C:\\Users\\estel\\STAGE\\Dataset\\predictions_EFFNET.csv")

# Obtenir les données ROC pour chaque modèle
fpr_cnn, tpr_cnn, roc_auc_cnn = get_roc_data(df_cnn, label_col='y_true', prob_col='y_pred_prob_melanoma')
fpr_efficientnet, tpr_efficientnet, roc_auc_efficientnet = get_roc_data(df_efficientnet, label_col='y_true', prob_col='y_pred_prob_effnet_melanoma')

# Tracer les courbes ROC des modèles CNN et EfficientNetB0
plt.figure(figsize=(10, 8))
plt.plot(fpr_cnn, tpr_cnn, lw=2, label=f'CNN (AUC = {roc_auc_cnn:0.2f})')
plt.plot(fpr_efficientnet, tpr_efficientnet, lw=2, label=f'EfficientNetB0 (AUC = {roc_auc_efficientnet:0.2f})')

# Ajouter la diagonale
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Limites et légende
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbes ROC des modèles CNN et EfficientNetB0')
plt.legend(loc='lower right')

# Afficher le graphe
plt.show()
