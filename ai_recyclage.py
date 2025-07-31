import os
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime

# === Configuration ===
SEUIL_CONFIANCE = 0.65  # seuil en dessous duquel on enregistre les images
RETRAIN_DIR = "a_reentrainer"  # dossier local de stockage
os.makedirs(RETRAIN_DIR, exist_ok=True)
LOG_FILE = os.path.join(RETRAIN_DIR, "log.csv")

# === Mots-clés associés aux catégories recyclables ===
plastique_keywords = ['bottle', 'plastic', 'container', 'cup']
metal_keywords = ['can', 'metal', 'tin', 'aluminum']
organique_keywords = ['banana', 'apple', 'fruit', 'vegetable', 'food']

# === Charger le modèle MobileNetV2 pré-entraîné (ImageNet) ===
model = tf.keras.applications.MobileNetV2(weights='imagenet')
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

# === Initialiser la webcam ===
cap = cv2.VideoCapture(0)  # 0 = première webcam

def save_frame(frame, prediction, confidence):
    """Enregistrer une image et les infos dans un fichier CSV"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prediction}_{int(confidence * 100)}_{timestamp}.jpg"
    filepath = os.path.join(RETRAIN_DIR, filename)
    cv2.imwrite(filepath, frame)

    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp},{prediction},{confidence:.2f},{filename}\n")

# === Boucle principale ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prétraitement de l'image
    img = cv2.resize(frame, (224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Prédiction IA
    predictions = model.predict(img_array)
    decoded = decode_predictions(predictions, top=1)[0][0]  # top 1 résultat
    label = decoded[1]  # nom de l'objet
    confidence = decoded[2]  # score de confiance

    # === Vérification de la confiance ===
    if confidence < SEUIL_CONFIANCE:
        # Cas incertain → sauvegarde
        status = "Non recyclable (Incertitude)"
        color = (0, 0, 255)
        save_frame(frame, label, confidence)

    else:
        # On vérifie le type de déchet
        label_lower = label.lower()
        if any(word in label_lower for word in plastique_keywords):
            status = "♻️ Recyclable (Plastique)"
            color = (0, 255, 0)
        elif any(word in label_lower for word in metal_keywords):
            status = "♻️ Recyclable (Métal)"
            color = (255, 255, 0)
        elif any(word in label_lower for word in organique_keywords):
            status = "♻️ Recyclable (Organique)"
            color = (0, 128, 255)
        else:
            status = "🗑️ Non recyclable"
            color = (100, 100, 100)

    # Affichage
    text = f"{label} ({confidence * 100:.1f}%)"
    color = (0, 255, 0) if confidence >= SEUIL_CONFIANCE else (0, 0, 255)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Sauvegarde si incertain
    if confidence < SEUIL_CONFIANCE:
        save_frame(frame, label, confidence)

    # Afficher l'image en live
    cv2.imshow("IA Recyclage - Appuyez sur Q pour quitter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Nettoyage
cap.release()
cv2.destroyAllWindows()
