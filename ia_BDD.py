import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from datetime import datetime

from tensorflow.python.keras.saving.save import load_model
from tensorflow.keras.models import load_model
#model = load_model("modele_recyclage.keras")



# === Configuration ===
image_size = (224, 224)
batch_size = 16
dataset_path = "dataset"
test_path = "test_images"
threshold_precision = 0.6
dossier_incertains = "incertains"
LOG_FILE = os.path.join(dossier_incertains, "log.csv")
model_path = "modele_recyclage.keras"

#cette ligne me permet de charger le modele deja entrainer qui est enregistrer dans le fichier .h5

#model= load_model(model_path)

#je cree un dictionnaire de classe pour garder manuellement au cas ou le modele ne'est pas connecter pour generer


# # === Création du générateur d'image pour entraînement ===
# #datagen = ImageDataGenerator(validation_split=0.2, rescale=1. / 255)
#
# #train_generator = datagen.flow_from_directory(
#    # dataset_path,
#     #target_size=image_size,
#     #batch_size=batch_size,
#     #class_mode="categorical",
#     #subset="training"
# #)
#
# #val_generator = datagen.flow_from_directory(
#     #dataset_path,
#     #target_size=image_size,
#     #batch_size=batch_size,
#     #class_mode="categorical",
#     subset="validation"
# )
#
# === Entraînement du modèle ===
# base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
# base_model.trainable = False
#
# model = models.Sequential([
#     base_model,
#     layers.GlobalAveragePooling2D(),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.3),
#     layers.Dense(train_generator.num_classes, activation='softmax')
# ])
#
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_generator, validation_data=val_generator, epochs=10)
# model.save(model_path)
#
# # === Dictionnaire des classes (nom ↔ index) ===
# class_indices = train_generator.class_indices
# index_to_class = {v: k for k, v in class_indices.items()}


reentrainer =not os.path.exists(model_path)

if reentrainer:
    # === ENTRAÎNEMENT DU MODÈLE ===
    datagen = ImageDataGenerator(validation_split=0.2, rescale=1. / 255)

    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    val_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, validation_data=val_generator, epochs=10)
    model.save("modele_recyclage.keras")  # format natif plus robuste


    class_indices = train_generator.class_indices
    index_to_class = {v: k for k, v in class_indices.items()}

else:

    model= load_model(model_path)

    index_to_class = {
        0: "metal",
        1: "no_recycle",
        2: "organique",
        3: "plastic"
    }

# === Préparation des dossiers ===
os.makedirs(dossier_incertains, exist_ok=True)

# === Fonction pour sauvegarder image incertaine ===
def save_frame(frame, label, confidence):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{int(confidence * 100)}_{timestamp}.jpg"
    filepath = os.path.join(dossier_incertains, filename)
    cv2.imwrite(filepath, frame)

    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp},{label},{confidence:.2f},{filename}\n")

# === Démarrage webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Impossible d'accéder à la caméra.")
    exit()

print("✅ Caméra activée. Appuie sur [ESPACE] pour analyser ou [Q] pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Erreur de lecture de la caméra.")
        break

    # Affichage caméra en direct
    cv2.imshow("IA Recyclage - Webcam", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Quitter avec Q
        break

    elif key == 32:  # Touche ESPACE (code ASCII 32)
        print("🔍 Analyse en cours...")

        # === Traitement de l'image ===
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize(image_size)
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        # === Prédiction IA ===
        prediction = model.predict(img_array)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        label = index_to_class[predicted_class]

        # === Affichage du résultat sur l'image ===
        color = (0, 255, 0) if confidence >= threshold_precision else (0, 0, 255)
        text = f"{label} ({confidence * 100:.1f}%)"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Afficher le résultat dans une fenêtre distincte
        cv2.imshow("Résultat IA", frame)

        # Sauvegarde si incertitude
        if confidence < threshold_precision:
            save_frame(frame, label, confidence)
            print(f"⚠️ Image incertaine enregistrée ({confidence:.2f})")

        else:
            print(f"✅ Résultat : {label} ({confidence*100:.1f}%)")



# Libération
cap.release()
cv2.destroyAllWindows()
