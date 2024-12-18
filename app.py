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

app = Flask(__name__)

# Definir o caminho para a pasta `static/uploads`
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Caminho dos arquivos do modelo de predição de landmarks e reconhecimento facial
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"  # Caminho para o modelo de reconhecimento facial

# Inicializando o detector facial, o preditor de pontos e o modelo de reconhecimento facial
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_recognition_model = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

# Função para verificar se a extensão do arquivo é permitida
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Função para gerar o embedding (vetor de características) de uma imagem
def get_face_embedding(image_path):
    # Carregar imagem
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar rostos
    faces = detector(gray)
    
    if len(faces) == 0:
        return None

    # Selecionar o primeiro rosto detectado
    face = faces[0]

    # Obter os pontos de referência (landmarks)
    landmarks = predictor(gray, face)

    # Extrair as características faciais (embeddings) usando o modelo de reconhecimento facial
    face_descriptor = np.array(face_recognition_model.compute_face_descriptor(img, landmarks))
    
    return face_descriptor

# Função para comparar dois embeddings usando distância euclidiana
def compare_embeddings_euclidean(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)  # Distância Euclidiana

# Threshold para considerar a mesma pessoa usando distância euclidiana
EUCLIDEAN_THRESHOLD = 0.5  # Ajuste o valor conforme necessário

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

        # Comparar a imagem carregada com a foto base
        base_image_embedding = get_face_embedding('static/uploads/base_image.jpg')  # A foto base está salva na pasta uploads
        new_image_embedding = get_face_embedding(upload_path)

        if base_image_embedding is None or new_image_embedding is None:
            return "Não foi possível detectar um rosto em uma das imagens."

        # Comparar os embeddings usando distância euclidiana
        similarity = compare_embeddings_euclidean(base_image_embedding, new_image_embedding)

        # Se a similaridade for abaixo do threshold, consideramos que as faces são as mesmas
        if similarity < EUCLIDEAN_THRESHOLD:
            result = "As pessoas são as mesmas!"
        else:
            result = "As pessoas são diferentes."

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

        # Gerar nome único para a imagem capturada
        filename = f"image_captured.jpg"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Salvar a imagem como arquivo
        image = Image.open(BytesIO(img_data))
        image.save(upload_path)

        # Comparar a imagem capturada com a foto base
        base_image_embedding = get_face_embedding('static/uploads/base_image.jpg')
        new_image_embedding = get_face_embedding(upload_path)

        if base_image_embedding is None or new_image_embedding is None:
            return "Não foi possível detectar um rosto em uma das imagens."

        # Comparar os embeddings usando distância euclidiana
        similarity = compare_embeddings_euclidean(base_image_embedding, new_image_embedding)

        # Se a similaridade for abaixo do threshold, consideramos que as faces são as mesmas
        if similarity < EUCLIDEAN_THRESHOLD:
            result = "As pessoas são as mesmas!"
        else:
            result = "As pessoas são diferentes."

        return render_template('uploaded_image.html', filename=filename, result=result)
    return 'Erro ao capturar a imagem'

@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return render_template('uploaded_image.html', filename=filename)

if __name__ == '__main__':
    # Criar a pasta `static/uploads` se não existir
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)
