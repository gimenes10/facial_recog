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

# Configuração da pasta de uploads (dentro de static para ser servida como arquivo estático)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Carregar o classificador Haar Cascade para detecção facial
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Threshold para considerar que a face é a mesma (usando correlação: valores próximos de 1 indicam alta similaridade)
HIST_THRESHOLD = 0.5  # ajuste conforme testes

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_face_histogram(image_path):
    """
    Carrega a imagem, converte para escala de cinza, detecta o rosto e extrai seu histograma normalizado.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    # Seleciona o primeiro rosto detectado
    (x, y, w, h) = faces[0]
    roi = gray[y:y+h, x:x+w]
    hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist

def compare_histograms(hist1, hist2):
    """
    Compara dois histogramas usando o método de correlação (HISTCMP_CORREL).
    Valores próximos de 1.0 indicam alta similaridade.
    """
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def similarity_percentage(correlation):
    """
    Converte o valor de correlação (0 a 1) em porcentagem.
    """
    return correlation * 100

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

        # Extrai histogramas da imagem base e da imagem nova
        base_hist = get_face_histogram('static/uploads/base_image.jpg')
        new_hist = get_face_histogram(upload_path)
        if base_hist is None or new_hist is None:
            return "Não foi possível detectar um rosto em uma das imagens."

        # Compara os histogramas
        correlation = compare_histograms(base_hist, new_hist)
        sim_percent = similarity_percentage(correlation)

        if correlation >= HIST_THRESHOLD:
            result = f"Rosto reconhecido: Similaridade {sim_percent:.1f}%"
        else:
            result = f"Pessoa não encontrada: Similaridade {sim_percent:.1f}%"

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

        base_hist = get_face_histogram('static/uploads/base_image.jpg')
        new_hist = get_face_histogram(upload_path)
        if base_hist is None or new_hist is None:
            return "Não foi possível detectar um rosto na imagem capturada."

        correlation = compare_histograms(base_hist, new_hist)
        sim_percent = similarity_percentage(correlation)

        if correlation >= HIST_THRESHOLD:
            result = f"Rosto reconhecido: Similaridade {sim_percent:.1f}%"
        else:
            result = f"Pessoa não encontrada: Similaridade {sim_percent:.1f}%"

        return render_template('uploaded_image.html', filename=filename, result=result)
    return 'Erro ao capturar a imagem'

@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return render_template('uploaded_image.html', filename=filename)

@app.route('/list_users')
def list_users():
    users = []  # Se desejar, pode integrar com o banco de dados; aqui, apenas um placeholder.
    html = "<h1>Usuários Cadastrados</h1><ul>"
    for user in users:
        html += f"<li>ID: {user['id']} | Nome: {user['nome']}, CPF: {user['cpf']}, Foto: {user['foto_path']}</li>"
    html += "</ul><a href='/'>Voltar</a>"
    return html

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
