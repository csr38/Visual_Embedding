from flask import Flask, render_template, request, redirect, url_for, flash
from pymongo import MongoClient
from deepface import DeepFace
import os
import numpy as np
import cv2

# Configuración de la aplicación Flask
app = Flask(__name__)
app.secret_key = "secret_key"

# Configuración de MongoDB
MONGO_URI = "mongodb+srv://cesar:1jhhHaYVpUIBMmn2@cluster0.c7fw8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["visual"]  # Nombre de la base de datos
people_collection = db["people"]  # Nombre de la colección

# Carpeta para cargar imágenes
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Crear carpeta si no existe
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def extract_face(image_path):
    """
    Extrae el rostro de la imagen usando OpenCV.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None  # No se encontró rostro

    x, y, w, h = faces[0]  # Tomamos el primer rostro detectado
    face = image[y:y+h, x:x+w]
    face_resized = cv2.resize(face, (160, 160))  # Ajustar tamaño para modelos como Facenet
    return face_resized


# Ruta para registrar rostros (permitir múltiples fotos por persona)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form["name"]
        role = request.form["role"]  # funcionario o delincuente
        files = request.files.getlist("file")  # Cambiar para permitir múltiples archivos

        if files and name and role:
            for file in files:
                # Guardar la imagen temporalmente
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(file_path)

                # Extraer rostro
                face = extract_face(file_path)
                if face is None:
                    flash("No se detectó un rostro en la imagen.")
                    continue  # Si no se detecta rostro, se omite esta foto

                # Guardar el rostro extraído
                face_path = os.path.join(app.config["UPLOAD_FOLDER"], f"face_{file.filename}")
                cv2.imwrite(face_path, face)

                # Generar el embedding
                try:
                    embedding = DeepFace.represent(img_path=face_path, model_name='Facenet')[0]['embedding']
                except Exception as e:
                    flash(f"Error al generar el embedding: {str(e)}")
                    continue

                # Guardar en la base de datos
                people_collection.update_one(
                    {"name": name},
                    {
                        "$push": {"embeddings": embedding},
                        "$set": {"role": role}
                    },
                    upsert=True
                )

            flash(f"Las fotos de {name} fueron registradas como {role} exitosamente.")
            return redirect(url_for("index"))

    return render_template("index.html")



@app.route("/verify", methods=["GET", "POST"])
def verify():
    result = None
    similarity_percentage = None
    if request.method == "POST":
        file = request.files["file"]

        if file:
            # Guardar la imagen temporal
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Extraer rostro
            face = extract_face(file_path)
            if face is None:
                flash("No se detectó un rostro en la imagen.")
                return redirect(url_for("verify"))

            # Guardar el rostro extraído
            face_path = os.path.join(app.config["UPLOAD_FOLDER"], f"face_{file.filename}")
            cv2.imwrite(face_path, face)

            # Obtener el embedding del rostro subido
            try:
                input_embedding = DeepFace.represent(img_path=face_path, model_name='Facenet')[0]['embedding']
                input_embedding = np.array(input_embedding)  # Convertir a numpy array
            except Exception as e:
                flash(f"Error al procesar el rostro: {str(e)}")
                return redirect(url_for("verify"))

            # Buscar en la base de datos
            people = people_collection.find()
            min_distance = float('inf')
            identified_person = None

            for person in people:
                # Compara con todos los embeddings de esa persona
                for db_embedding in person["embeddings"]:
                    db_embedding = np.array(db_embedding)  # Asegúrate de que sea numpy array

                    # Calcular la distancia euclidiana
                    distance = np.linalg.norm(input_embedding - db_embedding)

                    if distance < min_distance:
                        min_distance = distance
                        identified_person = person

            # Umbral para determinar la similitud
            threshold = 15  # Umbral de distancia para considerar que es la misma persona

            if identified_person and min_distance < threshold:
                # Ajustar el cálculo del porcentaje de similitud
                similarity_percentage = max(0, (1 - min_distance / threshold) * 100)
                result = f"Identificado como {identified_person['role']}: {identified_person['name']} (distancia: {min_distance:.2f})"
            else:
                result = "Desconocido (posible cliente)"
                similarity_percentage = 0  # Si no se encuentra, el porcentaje es 0

    return render_template("verify.html", result=result, similarity_percentage=similarity_percentage)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
