# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:12:06 2024

@author: estel
"""

from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense,BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import seaborn as sns
import numpy as np
from PIL import Image 
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
import pandas as pd


class Classifier:
    
         
    def prepocessing(self, train_dir, test_dir, input_shape=(256, 256, 3)):
        self.input_shape = input_shape  
        self.image_size = (self.input_shape[0],self.input_shape[1])
        
        # chargement des données et repartition du jeux  de données de train_dir en 80% pour l'entrainement et 20% pour la validation
        train_data = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            validation_split=0.2
            )
            
        test_data = ImageDataGenerator(rescale=1./255)
                
        training_set = train_data.flow_from_directory(train_dir, 
                                                      target_size=self.image_size,
                                                      batch_size=32, 
                                                      shuffle=True,
                                                      class_mode="binary",
                                                      subset="training",
                                                      classes={'NEV': 0, 'MEL': 1})#Attribut les etiquettes correspondantes aux classes

        test_set = test_data.flow_from_directory(test_dir, 
                                                 target_size=self.image_size,
                                                 batch_size=32, 
                                                 shuffle=False,
                                                 class_mode="binary",
                                                 classes={'NEV': 0, 'MEL': 1})
        
        validation_set = train_data.flow_from_directory(
            train_dir,
            target_size=self.image_size,
            batch_size=32,
            shuffle=False,
            class_mode="binary",
            subset='validation',
            classes={'NEV': 0, 'MEL': 1}
            
        )
        
        return training_set, validation_set, test_set
        
    def build_model(self):
        model = Sequential()
        
        # Première couche Convolutionnelle 
        model.add(Convolution2D(filters=16, kernel_size=3,
                                strides=1, input_shape=self.input_shape,padding = 'Same', activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Deuxième couche Convolutionnelle 
        model.add(Convolution2D(filters=32, kernel_size=3,
                                strides=1, input_shape=self.input_shape, padding = 'Same',activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Troisième couche Convolutionnelle 
        model.add(Convolution2D(filters=64, kernel_size=3,
                                strides=1, input_shape=self.input_shape,padding = 'Same', activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        
        model.add(Convolution2D(filters=128, kernel_size=3,
                                strides=1, input_shape=self.input_shape, padding = 'Same',activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Convolution2D(filters=256, kernel_size=3,
                                strides=1, input_shape=self.input_shape,padding = 'Same', activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # model.add(Convolution2D(filters=256, kernel_size=3,
        #                         strides=1, input_shape=self.input_shape,padding = 'Same', activation="relu"))
        # model.add(BatchNormalization())
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # model.add(Convolution2D(filters=256, kernel_size=3,
        #                         strides=1, input_shape=self.input_shape,padding = 'Same', activation="relu"))
        # model.add(BatchNormalization())
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # #Dropout
        # model.add(Dropout(0.5))
        
        # Couche de flattening
        model.add(Flatten())
        
        # Couche dense
        # model.add(Dense(units=256, activation="relu"))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.25))
        # model.add(Dense(units=128, activation="relu"))
        # model.add(Dropout(0.25))
        model.add(Dense(units=64, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(units=1, activation="sigmoid"))
        
        return model
    
   
    
    def train(self, model_build, training_set, validation_set, numb_epochs):
        # Calculer les poids des classes pour le jeu d'entraînement
        class_labels = training_set.classes
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(class_labels), y=class_labels)
        class_weights = dict(enumerate(class_weights))
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='auc', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001)
        model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_auc', mode='max')


        
        # Compiler le modèle avec les poids de classes
        
        optimizer = Adam(learning_rate=0.01)
        model_build.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy', AUC(name='auc')])
        
        # Entraîner le modèle avec les poids de classes
        history = model_build.fit(
            training_set,
            epochs=numb_epochs,
            validation_data=validation_set,
            class_weight=class_weights, 
            callbacks=[early_stopping, reduce_lr, model_checkpoint]
        )
        
        
        return model_build, history
    
    def load_best_model(self, model_path):
        best_model = load_model(model_path)
        return best_model


    
    def predict(self, trained_model, test_set):
                
        y_pred_prob = trained_model.predict(test_set, steps=len(test_set))
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        y_true = test_set.classes
        filepaths = test_set.filepaths
        
        return y_pred_prob, y_pred,y_true,filepaths
    
    def save_prob_to_csv(self,filepaths, y_pred_prob, y_pred,y_true,predictions_csv_path):
        
        # Extraire le nom de l'image à partir du chemin
        image_names = [filepath.split("\\")[-1]for filepath in filepaths]

        df_cnn = pd.DataFrame({
        'nom':image_names,   
        'filepath': filepaths,
        'y_true': y_true,
        'y_pred_prob_melanoma': y_pred_prob[:, 0],  # Probabilité pour la seule classe (melanoma)
        'y_pred_prob_nevus': 1 - y_pred_prob[:, 0],  # Calcul de la probabilité pour la classe 'nevus'
        'y_pred': y_pred  # Classe prédite
       })

        # Sauvegarder dans un fichier CSV
        df_cnn.to_csv(predictions_csv_path, index=False)
            
    
    def evaluate(self, y_pred_prob, y_pred,y_true):

        # Matrice de confusion
        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Nevus', 'Melanoma'], yticklabels=['Nevus', 'Melanoma'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

        # Rapport de classification
        print(classification_report(y_true, y_pred, target_names=['Nevus', 'Melanoma']))

        # Courbe ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        roc_auc = roc_auc_score(y_true, y_pred_prob)
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
    
    def display_correctly_classified_images(self, filepaths, y_pred, y_true):
        correctly_classified_indices = np.where(y_pred == y_true)[0]

        # Affichage des images bien classées
        num_images_to_display = min(5, len(correctly_classified_indices))  # Afficher jusqu'à 5 images
        fig, axes = plt.subplots(2, num_images_to_display // 2, figsize=(15, 8))
    
        for i, ax in enumerate(axes.flat):
            if i < num_images_to_display:
                index = correctly_classified_indices[i]
                image = Image.open(filepaths[index])  # Charge l'image à partir du chemin
                label = 'Nevus' if y_true[index] == 0 else 'Melanoma'
                ax.imshow(image)
                ax.axis('off')
                ax.set_title(f'Predicted: {label}')
    
        plt.tight_layout()
        plt.show()         
             
     

if __name__ == "__main__":
    cnn_classifier = Classifier()
    train_dir = "C:\\Users\\estel\\STAGE\\Dataset\\NON_SEGMENTEES\\TRAIN"
    test_dir = "C:\\Users\\estel\\STAGE\\Dataset\\NON_SEGMENTEES\\TEST"
    input_shape = (256, 256, 3)
   
    numb_epochs=70
    
    model_path =  "C:\\Users\\estel\\STAGE\\best_model.keras"

    predictions_csv_path = "C:\\Users\\estel\\STAGE\\Dataset\\predictions_cnn.csv"


    # Importation et pré-traitement des données
    
    training_set, validation_set, test_set = cnn_classifier.prepocessing(train_dir, test_dir, input_shape)
    
    #Construction du modèle  
    model_build = cnn_classifier.build_model()

    # Entrainement du modèle
    trained_model,history = cnn_classifier.train(model_build, training_set, validation_set, numb_epochs)
    
    # Charger le meilleur modèle sauvegardé par ModelCheckpoint
    best_model = cnn_classifier.load_best_model(model_path)
    
    # Prediction 
    y_pred_prob, y_pred,y_true,filepaths= cnn_classifier.predict(best_model, test_set)
    
    #sauvegarde sous forme csv
    cnn_classifier.save_prob_to_csv(filepaths, y_pred_prob, y_pred, y_true, predictions_csv_path)

    #  Evaluation
    cnn_classifier.evaluate(y_pred_prob, y_pred,y_true)
    
    # STEP 6: Afficher les images bien classées
    correctly_classified_indices=cnn_classifier.display_correctly_classified_images(filepaths, y_pred, y_true)


    