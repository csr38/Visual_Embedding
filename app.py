from flask import Flask, render_template, request, redirect, url_for, flash, Response
from pymongo import MongoClient
from deepface import DeepFace
from mtcnn import MTCNN
import cv2
import os
import numpy as np

# Configuración de la aplicación Flask
app = Flask(__name__)
app.secret_key = "secret_key"

# Configuración de MongoDB
MONGO_URI = "mongodb+srv://cesar:1jhhHaYVpUIBMmn2@cluster0.c7fw8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["visual"]
people_collection = db["people"]

# Carpeta para almacenar videos
VIDEO_FOLDER = "videos"
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Inicializar MTCNN para la detección de rostros
detector = MTCNN()

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
        face_list.append((face_resized, (x, y, w, h)))

    return face_list

# Verificación de personas con embeddings almacenados en MongoDB
def verify_person(face_embedding):
    people = people_collection.find()
    min_distance = float('inf')
    identified_person = None
    threshold = 5.5  # Distancia umbral

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

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                faces = extract_faces(frame)
                for face, rect in faces:
                    try:
                        embedding = DeepFace.represent(face, model_name='Facenet', enforce_detection=False)[0]['embedding']
                        name, role, distance = verify_person(np.array(embedding))

                        if role == "delincuente":
                            color = (0, 0, 255)  # Rojo
                            label = f"Delincuente: {name}"
                        elif role == "trabajador":
                            color = (0, 165, 255)  # Naranja
                            label = f"Trabajador: {name}"
                        else:
                            color = (0, 255, 0)  # Verde
                            label = f"Desconocido"

                        x, y, w, h = rect
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    except Exception as e:
                        print(f"Error al procesar el rostro: {str(e)}")

                ret, jpeg = cv2.imencode('.jpg', frame)
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
        name = request.form["name"]
        role = request.form["role"]

        for file in files:
            image_path = os.path.join("static/uploads", file.filename)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            file.save(image_path)

            img = cv2.imread(image_path)
            faces = extract_faces(img)

            if faces:
                for face, _ in faces:
                    embedding = DeepFace.represent(face, model_name='Facenet', enforce_detection=False)[0]['embedding']

                    person = people_collection.find_one({"name": name})
                    if person:
                        people_collection.update_one({"_id": person["_id"]}, {"$push": {"embeddings": embedding}})
                        flash(f"Nuevo embedding agregado para {name}.")
                    else:
                        people_collection.insert_one({"name": name, "role": role, "embeddings": [embedding]})
                        flash(f"Rostro de {name} registrado correctamente.")
            else:
                flash("No se detectaron rostros en las imágenes.")

        return redirect(url_for("add_faces"))

    return render_template("add_faces.html")

# Ejecutar la aplicación Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
