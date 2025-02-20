from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import base64
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image
import sqlite3
import json

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Limiar de similaridade (pode ser ajustado conforme testes)
HIST_THRESHOLD = 0.5

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_face_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    # Converte para escala de cinza (pipeline original para manter compatibilidade com os embeddings cadastrados)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detecção de face com parâmetros originais
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    # Seleciona o primeiro rosto detectado (ou você pode optar pelo maior, se preferir)
    (x, y, w, h) = faces[0]
    roi = gray[y:y+h, x:x+w]
    # Calcula o histograma da região de interesse (ROI)
    hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).astype('float32')
    return hist

def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def similarity_percentage(correlation):
    return correlation * 100

def init_db():
    conn = sqlite3.connect('app2.db')
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
    conn = sqlite3.connect('app2.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, nome, cpf, foto_path, embedding FROM usuarios")
    rows = cursor.fetchall()
    conn.close()

    users = []
    for r in rows:
        uid, nome, cpf, foto_path, hist_json = r
        hist_list = json.loads(hist_json)
        hist_arr = np.array(hist_list, dtype=np.float32)
        users.append({'id': uid, 'nome': nome, 'cpf': cpf, 'foto_path': foto_path, 'embedding': hist_arr})
    
    return users

def user_exists(cpf):
    conn = sqlite3.connect('app2.db')
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM usuarios WHERE cpf = ?", (cpf,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def insert_user(nome, cpf, foto_path, embedding):
    if user_exists(cpf):
        raise ValueError(f"CPF {cpf} já cadastrado no banco de dados.")
    hist_json = json.dumps(embedding.tolist())
    conn = sqlite3.connect('app2.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO usuarios (nome, cpf, foto_path, embedding) VALUES (?, ?, ?, ?)",
                   (nome, cpf, foto_path, hist_json))
    conn.commit()
    conn.close()

def recognize_user(new_hist):
    users = get_all_users()
    best_user = None
    best_correlation = 0
    # Compara o histograma da imagem atual com os de cada usuário cadastrado
    for user in users:
        correlation = compare_histograms(new_hist, user['embedding'])
        if correlation > best_correlation:
            best_correlation = correlation
            best_user = user
    # Retorna o usuário apenas se a similaridade ultrapassar o limiar definido
    if best_correlation >= HIST_THRESHOLD:
        return best_user, best_correlation
    return None, best_correlation

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

        new_hist = get_face_embedding(upload_path)
        if new_hist is None:
            return "Não foi possível detectar um rosto na imagem enviada."

        recognized_user, correlation = recognize_user(new_hist)
        sim_percent = similarity_percentage(correlation)

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

        new_hist = get_face_embedding(upload_path)
        if new_hist is None:
            return "Não foi possível detectar um rosto na imagem capturada."

        recognized_user, correlation = recognize_user(new_hist)
        sim_percent = similarity_percentage(correlation)

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
