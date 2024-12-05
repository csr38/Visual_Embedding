from flask import Flask, render_template, request, redirect, url_for, flash, Response
from pymongo import MongoClient
from deepface import DeepFace
from mtcnn import MTCNN
import cv2
import os
import numpy as np
import requests
import tempfile
from datetime import datetime

# Configuración de la aplicación Flask
app = Flask(__name__)
app.secret_key = "secret_key"

# Configuración de MongoDB
MONGO_URI = "mongodb+srv://cesar:1jhhHaYVpUIBMmn2@cluster0.c7fw8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["visual"]
people_collection = db["people"]

# Inicializar MTCNN para detección de rostros
detector = MTCNN()

# Carpeta para almacenar videos
VIDEO_FOLDER = "videos"
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Token Telegram
BOT_TOKEN = "7385432946:AAEcusX5tZ3uH_D-1PN_KHg-RM9y4Pm9b64"
# ID Telegram
CHAT_ID = "6536885057"

# Función para enviar una imagen y un mensaje a Telegram
def send_telegram_notification_with_image(message, image, role, features):
    try:
        # Guardar la imagen en un archivo temporal
        _, img_path = tempfile.mkstemp(suffix='.jpg')
        cv2.imwrite(img_path, image)

        # Obtener la hora actual
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if role == "pi":
            message = f"<b><font color='red'>{message}</font></b>"
        elif role == "trabajador":
            message = f"<b><font color='green'>{message}</font></b>"

        # Incluir la descripción en el mensaje y la hora
        message += f"\nDescripción: {features}\nHora: {current_time}"
        
        # Enviar mensaje con la imagen
        with open(img_path, 'rb') as img_file:
            response = requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                data={"chat_id": CHAT_ID, "caption": message},
                files={"photo": img_file}
            )
            
        # Eliminar archivo temporal después de enviarlo
        os.remove(img_path)
        
        if response.status_code != 200:
            print(f"Error al enviar mensaje: {response.json()}")
        else:
            print(f"Mensaje enviado: {message}")
    except Exception as e:
        print(f"Error enviando notificación: {e}")


# Función para detectar y extraer rostros en una imagen
def extract_faces(frame):
    if frame is None:
        print("Frame es None")
        return []

    faces = detector.detect_faces(frame)
    face_list = []

    for face in faces:
        x, y, w, h = face['box']
        face_roi = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (160, 160))
        face_list.append((face_resized, (x, y, w, h), face_roi))  # Agregar el rostro original (face_roi) a la lista

    return face_list

# Función para verificar personas en la base de datos usando embeddings
def verify_person(face_embedding):
    people = people_collection.find()
    min_distance = float('inf')
    identified_person = None
    threshold = 1.48  # Ajustar el umbral según la precisión deseada

    for person in people:
        for db_embedding in person["embeddings"]:
            db_embedding = np.array(db_embedding)
            distance = np.linalg.norm(face_embedding - db_embedding)
            if distance < min_distance:
                min_distance = distance
                identified_person = person

    if identified_person and min_distance < threshold:
        return identified_person["name"], identified_person["role"], min_distance
    return None, None, None

# Generador de video en vivo con detección de rostros
def gen_video():
    video_files = [os.path.join(VIDEO_FOLDER, f) for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi'))]

    while True:
        for video_file in video_files:
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                print(f"No se pudo abrir el archivo de video: {video_file}")
                continue

            frame_counter = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_counter += 1
                if frame_counter % 7 != 0:  # Procesar cada 5 cuadros
                    continue

                # Reducir resolución para detección y transmisión
                frame_resized = cv2.resize(frame, (640, 360))
                faces = extract_faces(frame_resized)

                for face_resized, rect, face_roi in faces:
                    try:
                        embedding = DeepFace.represent(face_resized, model_name='Facenet', enforce_detection=False)[0]['embedding']
                        role, features, distance = verify_person(np.array(embedding))

                        # Etiquetar rostros
                        x, y, w, h = rect
                        color, label = (0, 255, 0), "Desconocido"
                        if role == "pi":
                            color, label = (0, 0, 255), f"Persona de interes: {features}"
                            # Enviar notificación con la imagen del rostro
                            send_telegram_notification_with_image(f"Alerta: Posible persona de interes detectado", face_roi)
                        elif role == "trabajador":
                            color, label = (0, 165, 255), f"Trabajador: {features}"

                        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame_resized, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    except Exception as e:
                        print(f"Error al procesar el rostro: {str(e)}")

                # Enviar cuadro procesado al cliente
                ret, jpeg = cv2.imencode('.jpg', frame_resized)
                if not ret:
                    continue

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

            cap.release()

# Ruta para mostrar el video en vivo
@app.route("/video_feed")
def video_feed():
    return Response(gen_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Ruta principal de la aplicación
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Ruta para agregar rostros a la base de datos
@app.route("/add_faces", methods=["GET", "POST"])
def add_faces():
    if request.method == "POST":
        files = request.files.getlist("images")
        role = request.form["role"]
        features = request.form["features"]

        for file in files:
            image_path = os.path.join("static/uploads", file.filename)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            file.save(image_path)

            img = cv2.imread(image_path)
            faces = extract_faces(img)

            if faces:
                for face, _ in faces:
                    embedding = DeepFace.represent(face, model_name='Facenet', enforce_detection=False)[0]['embedding']
                    person = people_collection.find_one({"features": features})
                    if person:
                        people_collection.update_one({"_id": person["_id"]}, {"$push": {"embeddings": embedding}})
                        flash(f"Nuevo embedding agregado para las características proporcionadas.")
                    else:
                        people_collection.insert_one({"features": features, "role": role, "embeddings": [embedding]})
                        flash(f"Rostro registrado correctamente con las características proporcionadas.")
            else:
                flash("No se detectaron rostros en las imágenes.")

        return redirect(url_for("add_faces"))

    return render_template("add_faces.html")

# Ejecutar la aplicación Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
