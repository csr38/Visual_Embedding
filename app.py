from flask import Flask, render_template, request, redirect, url_for, flash
from pymongo import MongoClient
from deepface import DeepFace
import os
import numpy as np

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

# Ruta para registrar rostros
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form["name"]
        role = request.form["role"]  # funcionario o delincuente
        file = request.files["file"]

        if file and name and role:
            # Guardar la imagen
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Generar el embedding
            try:
                embedding = DeepFace.represent(img_path=file_path, model_name='Facenet')[0]['embedding']
            except Exception as e:
                flash(f"Error al generar el embedding: {str(e)}")
                return redirect(url_for("index"))

            # Guardar en la base de datos
            people_collection.insert_one({
                "name": name,
                "role": role,
                "embedding": embedding
            })

            flash(f"{name} registrado como {role} exitosamente.")
            return redirect(url_for("index"))

    return render_template("index.html")


# Ruta para verificar rostros
@app.route("/verify", methods=["GET", "POST"])
def verify():
    result = None
    if request.method == "POST":
        file = request.files["file"]

        if file:
            # Guardar la imagen temporal
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Obtener el embedding de la imagen subida
            try:
                input_embedding = DeepFace.represent(img_path=file_path, model_name='Facenet')[0]['embedding']
            except:
                flash("No se pudo procesar el rostro.")
                return redirect(url_for("verify"))

            # Buscar en la base de datos
            people = people_collection.find()
            min_distance = float('inf')
            identified_person = None

            for person in people:
                db_embedding = np.array(person["embedding"])
                distance = np.linalg.norm(np.array(input_embedding) - db_embedding)

                if distance < min_distance:
                    min_distance = distance
                    identified_person = person

            if identified_person and min_distance < 0.6:
                result = f"Identificado como {identified_person['role']}: {identified_person['name']} (distancia: {min_distance:.2f})"
            else:
                result = "Desconocido (posible cliente)"

    return render_template("verify.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
