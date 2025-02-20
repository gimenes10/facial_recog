import os
from app import get_face_embedding, insert_user, delete_user, init_db
from app2 import get_face_embedding, insert_user, init_db
from app3 import get_face_embedding, insert_user, init_db


def register_user(nome, cpf, foto_path):
    # Verifica se a imagem existe
    if not os.path.exists(foto_path):
        print(f"Arquivo de imagem não encontrado: {foto_path}")
        return

    # Gera o embedding da imagem
    embedding = get_face_embedding(foto_path)
    if embedding is None:
        print("Não foi possível detectar um rosto na imagem fornecida.")
        return

    # Insere o usuário no banco
    try:
        insert_user(nome, cpf, foto_path, embedding)
        print(f"Usuário '{nome}' (CPF: {cpf}) inserido com sucesso no banco.")
    except Exception as e:
        print(f"Erro ao inserir usuário no banco: {e}")

# Exemplo de uso:
if __name__ == "__main__":
    init_db()
    # Exemplo: Registrar um usuário manualmente
    nome = "Guilherme Gimenes"
    cpf = "44167932865"
    foto = "static/uploads/base_image.jpg"  # Ajuste para o caminho da imagem do usuário
    
    register_user(nome, cpf, foto)

    nome = "Gabriel Mascagni"
    cpf = "12345678901"
    foto = "static/uploads/base_image2.jpg"  # Ajuste para o caminho da imagem do usuário
    
    register_user(nome, cpf, foto)

    nome = "Luiz Casella"
    cpf = "12345678902"
    foto = "static/uploads/base_image3.jpg"  # Ajuste para o caminho da imagem do usuário
    
    register_user(nome, cpf, foto)

    nome = "João Pedro (Val)"
    cpf = "12345678903"
    foto = "static/uploads/base_image4.jpg"  # Ajuste para o caminho da imagem do usuário
    
    register_user(nome, cpf, foto)
