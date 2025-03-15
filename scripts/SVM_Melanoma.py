# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 18:23:02 2024

@author: estel
"""
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.stats import entropy, kurtosis, skew
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE


class Classifier:
    
    def load_images(self, image_dir,image_dir_hsv):
        
         
         image_paths = glob.glob(image_dir +'/*/*.jpg')  
         
         for path in image_paths:
               img = cv2.imread(path)
               img_name= path.split("\\")[-1]
               print(img_name)
               img_path= image_dir_hsv+ "\\"+ path.split("\\")[-2] + "\\"+ img_name
               
               if img is not None:  # Vérifier si l'image a été chargée avec succès
                   hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                   cv2.imwrite(img_path, hsv_img)
                   print(f"Saved HSV image to {img_path}")
                   
                   
    
               else:
                   print(f"Impossible de charger l'image : {path}")
           
     
         return     
    
           
    def calculate_features(self, img, img_hsv):
          features = {}
          # RGB components
          for i, color in enumerate(['R', 'G', 'B']):
              channel = img[:, :, i]
              features[f'{color}_entropy'] = entropy(channel.ravel())
              features[f'{color}_std'] = np.std(channel)
              features[f'{color}_kurtosis'] = kurtosis(channel.ravel(), fisher=False)
              features[f'{color}_skewness'] = skew(channel.ravel())
             
             
              print(f"{color} - Entropy: {features[f'{color}_entropy']}, Std: {features[f'{color}_std']}, Kurtosis: {features[f'{color}_kurtosis']}, Skewness: {features[f'{color}_skewness']}")
    
          # HSV components
          for i, color in enumerate(['H', 'S', 'V']):
              channel = img_hsv[:, :, i]
              features[f'{color}_entropy'] = entropy(channel.ravel())
              features[f'{color}_std'] = np.std(channel)
              features[f'{color}_kurtosis'] = kurtosis(channel.ravel(), fisher=False)
              features[f'{color}_skewness'] = skew(channel.ravel())
             
              print(f"{color} - Entropy: {features[f'{color}_entropy']}, Std: {features[f'{color}_std']}, Kurtosis: {features[f'{color}_kurtosis']}, Skewness: {features[f'{color}_skewness']}")
     
         
          return features
      
    def extract_features(self, image_dir_hsv, label):
         image_paths = glob.glob(image_dir_hsv + '/*/*.jpg')  
         features_list = []
         
         for path in image_paths:
             print(f"Extracting features from {path}")  # vérifier les chemins des images converties
             img_hsv = cv2.imread(path)
             if img_hsv is not None:
                 features = self.calculate_features(img_hsv, img_hsv)
                 features['image_path'] = path 
                 features['nom'] = path.split("\\")[-1] 
                 features['label'] = label
                 features_list.append(features)
             else:
                 print(f"Impossible de charger l'image : {path}")
         
         return features_list
     
    def prepare_data(self, features, label_mapping):
          data = []
          labels = []
          for feature in features:
             if 'image_path' in feature:
                 # Exclure les colonnes non numériques telles que 'image_path', 'nom', et 'label'
                 feature_values = [v for k, v in feature.items() if k not in ['image_path', 'nom', 'label']]
                 
                 image_path = feature['image_path']
                 print(f"Chemin de l'image : {image_path}")  # Affichage de débogage
                 
                 label = label_mapping[image_path.split('\\')[-2]]
                 data.append(feature_values)
                 labels.append(label)
             else:
                 print("La clé 'image_path' est manquante dans ce dictionnaire :", feature)
    
          return np.array(data, dtype=float), np.array(labels)     
      
         
      
    def save_features_to_csv(self, train_features, test_features, y_test, y_pred, y_pred_proba, train_csv_path, test_csv_path):
         # Sauvegarde des données d'entraînement
         train_df = pd.DataFrame(train_features)
         
         ##Réorganiser les colonnes pour les données d'entraînement
         train_columns_order = ['nom', 'image_path', 'label'] + [col for col in train_df.columns if col not in ['nom', 'image_path', 'label']]
         train_df = train_df[train_columns_order]
         
         train_df.rename(columns={"label": "set"}, inplace=True)
         train_df.to_csv(train_csv_path, index=False)
         print(f"Données d'entraînement sauvegardées dans {train_csv_path}")
         
         # Sauvegarde des données de test
         for i, feature in enumerate(test_features):
             feature['true_label'] = y_test[i]
             feature['pred_prob'] = y_pred_proba[i]
             feature['pred_label'] = y_pred[i]
         
         test_df = pd.DataFrame(test_features)
         
         ## Réorganiser les colonnes pour les données de test
         test_columns_order = ['nom', 'image_path', 'label'] + [col for col in test_df.columns if col not in ['nom', 'image_path', 'label', 'true_label', 'pred_prob', 'pred_label']] + ['true_label', 'pred_prob', 'pred_label']
         test_df = test_df[test_columns_order]
         test_df.rename(columns={"label": "set"}, inplace=True)
         
         
         test_df.to_csv(test_csv_path, index=False)
         test_df.to_csv(test_csv_path, index=False)
         print(f"Données de test sauvegardées dans {test_csv_path}")
         
    def train(self, X_train, y_train, class_weight='balanced'):
        #   Calcul du rapport de classe
        class_counts = np.bincount(y_train)
        class_weights = {0: 1, 1: class_counts[0] / class_counts[1]}  # 0 pour Nevus, 1 pour Melanome

        #Équilibrage des données
        smote = SMOTE()
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
       
        # Normalisation des données
        scaler = StandardScaler()
        X_train_resampled = scaler.fit_transform(X_train_resampled)

        # Normalisation des données
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Entrainement du modèle SVM
        svm = SVC(kernel='rbf', class_weight=class_weight, probability=True)  # kernel peut être ajusté (rbf, linear, poly, etc.)
        svm.fit(X_train_scaled, y_train)
        return svm, scaler,class_weights

    def predict(self, model, scaler, X_test, image_paths):
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Associe les chemins d'image aux probabilités prédites
        results = [{'image_path': path, 'probability': prob} for path, prob in zip(image_paths, y_pred_proba)]
        return y_pred, y_pred_proba, results

    def evaluate(self, y_test, y_pred, y_pred_proba, class_weights=None):
        
        
     ## Matrice de confusion
     conf_matrix = confusion_matrix(y_test, y_pred)
     plt.figure(figsize=(8, 6))
     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Naevus', 'Melanome'], yticklabels=['Naevus', 'Melanome'])
     plt.xlabel('Classe pédite')
     plt.ylabel('Vraie classe')
     plt.title('Matrice de confusion')
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
     plt.xlabel('Spécifité')
     plt.ylabel('Sensibilité')
     
     plt.show()
    
     # Rapport de classification
     print(classification_report(y_test, y_pred, target_names=['Nevus', 'Melanoma']))
            
     # Calcul des métriques spécifiques
     tn, fp, fn, tp = conf_matrix.ravel()
     sensitivity = tp / (tp + fn)
     specificity = tn / (tn + fp)
    
     print(f'Sensitivity: {sensitivity:.2f}')
     print(f'Specificity: {specificity:.2f}')
     print(f'ROC AUC: {roc_auc:.2f}')
 
if __name__ == "__main__":
    svm_classifier = Classifier()
    train_dir = "C:\\Users\\estel\\STAGE\\Dataset\\SEG\\TRAIN"
    test_dir = "C:\\Users\\estel\\STAGE\\Dataset\\SEG\\TEST"
    
    train_dir_hsv = "C:\\Users\\estel\\STAGE\\Dataset\\HSV\\TRAIN"
    test_dir_hsv = "C:\\Users\\estel\\STAGE\\Dataset\\HSV\\TEST"

    train_csv_path = "C:\\Users\\estel\\STAGE\\Dataset\\features_train_svm.csv"
    test_csv_path = "C:\\Users\\estel\\STAGE\\Dataset\\features_test_svm.csv"
    
    # Extraction des features
    print("Extracting features for train images...")
    train_features = svm_classifier.extract_features(train_dir_hsv, label='train')
    print("Extracting features for test images...")
    test_features = svm_classifier.extract_features(test_dir_hsv, label='test')

    # Préparation des données
    label_mapping = {'NEV': 0, 'MEL': 1}
    X_train, y_train = svm_classifier.prepare_data(train_features, label_mapping)
    X_test, y_test = svm_classifier.prepare_data(test_features, label_mapping)
    
    # Entrainement du modèle SVM
    svm_model, scaler,class_weights=class_weights = svm_classifier.train(X_train, y_train)

    # Prédictions sur le jeu de test
    image_paths_test = [feature['image_path'] for feature in test_features]
    y_pred, y_pred_proba, predictions = svm_classifier.predict(svm_model, scaler, X_test, image_paths_test)

    # Enregistrement des résultats
    svm_classifier.save_features_to_csv(train_features, test_features, y_test, y_pred, y_pred_proba, train_csv_path, test_csv_path)

    # Évaluation du modèle
    svm_classifier.evaluate(y_test, y_pred, y_pred_proba,class_weights=class_weights)
