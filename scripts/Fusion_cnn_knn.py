# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 13:45:21 2024

@author: estel
"""

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


class Fusion:
    
    
    def fusionner_probabilites(self, effnet_csv_path, knn_csv_path, fusion_csv_path):
        # #Lecture des dataframes
        # df_cnn = pd.read_csv(cnn_csv_path)
        df_effnet = pd.read_csv(effnet_csv_path)
        df_knn = pd.read_csv(knn_csv_path)
        
        # Renommer les colonnes 
        # df_cnn.rename(columns={'y_true': 'true_label'}, inplace=True)
        df_effnet.rename(columns={'y_true': 'true_label'}, inplace=True)
        df_knn.rename(columns={'image_path': 'filepath'}, inplace=True)
        
        print("Colonnes df_cnn après renommage:")
        
        # print(df_cnn.columns)  
        
        print("Colonnes df_effnet après renommage:")
        print(df_effnet.columns)
        
        print(df_knn.columns)

        # Extraire le nom de l'image dans les dataframes
        # df_cnn['nom'] = df_cnn['filepath'].apply(lambda x: x.split("\\")[-1])
        df_effnet['nom'] = df_effnet['filepath'].apply(lambda x: x.split("\\")[-1])
        df_knn['nom'] = df_knn['filepath'].apply(lambda x: x.split("\\")[-1])
        
        
        #fusion des dataframes
        df_merged = pd.merge(df_knn, df_effnet, on='nom', how='inner')
        # df_merged = pd.merge(df_merged, df_knn, on='nom', how='inner')
        # S'assurer que la colonne 'true_label' est présente
        df_merged['true_label'] = df_effnet['true_label']
   
        # Vérifier le résultat de la fusion
        print("DataFrame Fusionné:")
        print(df_merged.head())

        # Calculer la moyenne des probabilités pour chaque classe
        df_merged['prob_melanoma'] = (df_merged['y_pred_prob_effnet_melanoma'] +
                                      df_merged['pred_prob'])/ 2
                                         
    
        
        # Prédire la classe finale en utilisant la probabilité moyenne pour melanoma
        df_merged['label_pred'] = (df_merged['prob_melanoma'] > 0.5).astype(int)
    
        #Sélectionner les colonnes souhaitées
        df_final = df_merged[['nom', 'true_label','pred_prob', 'prob_melanoma', 'label_pred']]
   
       # Sauvegarder les résultats dans un fichier CSV
        df_final.to_csv(fusion_csv_path, index=False)
        print(f"Résultats fusionnés sauvegardés dans {fusion_csv_path}")
        
        return df_final
 
    
 
    
    def evaluate(self, y_test, y_pred, y_pred_proba):
        
        
        
        # Matrice de confusion
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Nevus', 'Melanoma'], yticklabels=['Nevus', 'Melanoma'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

        # Rapport de classification
        print(classification_report(y_test, y_pred, target_names=['Nevus', 'Melanoma']))

        # Courbe ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

        # Calcul des métriques spécifiques
        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print(f'Sensitivity: {sensitivity:.2f}')
        print(f'Specificity: {specificity:.2f}')
        print(f'ROC AUC: {roc_auc:.2f}')
if __name__ == "__main__":
    
    
    fusion_classifier= Fusion()
    
    # cnn_csv_path = "C:\\Users\\estel\\STAGE\\Dataset\\predictions_cnn.csv"
    effnet_csv_path = "C:\\Users\\estel\\STAGE\\Dataset\\predictions_EFFNET.csv"
    knn_csv_path = "C:\\Users\\estel\\STAGE\\Dataset\\features_test.csv"
    fusion_csv_path = "C:\\Users\\estel\\STAGE\\Dataset\\fusion_predictions.csv"
    
   
    df_final = fusion_classifier.fusionner_probabilites(effnet_csv_path, knn_csv_path, fusion_csv_path)
    
    # Évaluer les performances du modèle de fusion
    y_test = df_final['true_label']
    y_pred = df_final['label_pred']
    y_pred_proba = df_final['prob_melanoma']
    
    fusion_classifier.evaluate(y_test, y_pred, y_pred_proba)
