from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import dlib
import numpy as np
import base64
from scipy.spatial.distance import euclidean
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image
import sqlite3
import json

app = Flask(__name__)

# Definir o caminho para a pasta `static/uploads`
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Caminho dos arquivos do modelo de predição de landmarks e reconhecimento facial
PREDICTOR_PATH = "shape_predictor_68_face_landmarks_GTX.dat"
FACE_RECOGNITION_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"

# Inicializando o detector, preditor e modelo de reconhecimento facial
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_recognition_model = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

# Threshold para considerar as faces como iguais
EUCLIDEAN_THRESHOLD = 0.5  # se a distância for menor que esse valor, consideramos que é a mesma pessoa

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_face_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    face = faces[0]
    landmarks = predictor(gray, face)
    face_descriptor = np.array(face_recognition_model.compute_face_descriptor(img, landmarks))
    return face_descriptor

def compare_embeddings_euclidean(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

def similarity_percentage(distance, min_d=0.3, max_d=0.6):
    """
    Converte a distância euclidiana em uma porcentagem de similaridade.
    Se a distância for menor que min_d, retorna 100%.
    Se for maior que max_d, retorna 0%.
    Caso contrário, faz uma interpolação linear.
    """
    if distance < min_d:
        return 100.0
    elif distance > max_d:
        return 0.0
    else:
        return (1 - (distance - min_d) / (max_d - min_d)) * 100

# Funções de banco de dados (exemplo com SQLite)
def init_db():
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            cpf TEXT NOT NULL UNIQUE,
            foto_path TEXT NOT NULL,
            embedding TEXT NOT NULL
        );
    ''')
    conn.commit()
    conn.close()

def get_all_users():
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, nome, cpf, foto_path, embedding FROM usuarios")
    rows = cursor.fetchall()
    conn.close()

    users = []
    for r in rows:
        uid, nome, cpf, foto_path, embedding_json = r
        embedding_list = json.loads(embedding_json)
        embedding_arr = np.array(embedding_list, dtype=float)
        users.append({
            'id': uid,
            'nome': nome,
            'cpf': cpf,
            'foto_path': foto_path,
            'embedding': embedding_arr
        })
    return users

def user_exists(cpf):
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM usuarios WHERE cpf = ?", (cpf,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def insert_user(nome, cpf, foto_path, embedding):
    if user_exists(cpf):
        raise ValueError(f"CPF {cpf} já cadastrado no banco de dados.")
    embedding_json = json.dumps(embedding.tolist())
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO usuarios (nome, cpf, foto_path, embedding) VALUES (?, ?, ?, ?)",
                   (nome, cpf, foto_path, embedding_json))
    conn.commit()
    conn.close()

def delete_user(cpf):
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM usuarios WHERE cpf = ?", (cpf,))
    conn.commit()
    conn.close()

def recognize_user(new_embedding):
    users = get_all_users()
    recognized_user = None
    min_distance = float('inf')
    for user in users:
        dist = compare_embeddings_euclidean(new_embedding, user['embedding'])
        print(f"Distância para {user['nome']}: {dist}")  # Debug
        if dist < min_distance:
            min_distance = dist
            if dist < EUCLIDEAN_THRESHOLD:
                recognized_user = user
            else:
                recognized_user = None
    return recognized_user, min_distance

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        new_image_embedding = get_face_embedding(upload_path)
        if new_image_embedding is None:
            return "Não foi possível detectar um rosto na imagem enviada."

        recognized_user, distance = recognize_user(new_image_embedding)
        sim_percent = similarity_percentage(distance)

        if recognized_user:
            result = f"Rosto reconhecido: {recognized_user['nome']} - Similaridade: {sim_percent:.1f}%"
        else:
            result = f"Pessoa não encontrada no banco de dados - Similaridade: {sim_percent:.1f}%"

        return render_template('uploaded_image.html', filename=filename, result=result)
    return 'Arquivo inválido'

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/take_photo', methods=['POST'])
def take_photo():
    image_data = request.form.get('image')
    if image_data:
        image_data = image_data.split(',')[1]
        img_data = base64.b64decode(image_data)

        filename = "image_captured.jpg"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        image = Image.open(BytesIO(img_data))
        image.save(upload_path)

        new_image_embedding = get_face_embedding(upload_path)
        if new_image_embedding is None:
            return "Não foi possível detectar um rosto na imagem capturada."

        recognized_user, distance = recognize_user(new_image_embedding)
        sim_percent = similarity_percentage(distance)

        if recognized_user:
            result = f"Rosto reconhecido: {recognized_user['nome']} - Similaridade: {sim_percent:.1f}%"
        else:
            result = f"Pessoa não encontrada no banco de dados - Similaridade: {sim_percent:.1f}%"

        return render_template('uploaded_image.html', filename=filename, result=result)
    return 'Erro ao capturar a imagem'

@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return render_template('uploaded_image.html', filename=filename)

@app.route('/list_users')
def list_users():
    users = get_all_users()
    html = "<h1>Usuários Cadastrados</h1><ul>"
    for user in users:
        html += f"<li>ID: {user['id']} | Nome: {user['nome']}, CPF: {user['cpf']}, Foto: {user['foto_path']}</li>"
    html += "</ul><a href='/'>Voltar</a>"
    return html

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    init_db()
    app.run(debug=True)
