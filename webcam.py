import numpy as np
import face_recognition as fr
import cv2
from engine import get_rostos

# Carregar rostos conhecidos e seus nomes
rostos_conhecidos, nomes_dos_rostos = get_rostos()

# Captura de vídeo
video_capture = cv2.VideoCapture(0)

# Processar apenas cada X quadros para melhorar o desempenho
processar_este_quadro = True

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Falha ao capturar a imagem da webcam.")
        break

    if processar_este_quadro:
        # Reduzir o tamanho do quadro para 1/4 do tamanho original para processamento mais rápido
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Encontrar as localizações de rostos e encodings na imagem reduzida
        localizacao_dos_rostos = fr.face_locations(rgb_small_frame)
        rosto_desconhecidos = fr.face_encodings(rgb_small_frame, localizacao_dos_rostos)

        nomes_detectados = []

        for rosto_desconhecido in rosto_desconhecidos:
            resultados = fr.compare_faces(rostos_conhecidos, rosto_desconhecido)
            face_distances = fr.face_distance(rostos_conhecidos, rosto_desconhecido)

            if resultados:
                melhor_id = np.argmin(face_distances)
                if resultados[melhor_id]:
                    nome = nomes_dos_rostos[melhor_id]
                else:
                    nome = "Desconhecido"
                nomes_detectados.append(nome)
            else:
                nomes_detectados.append("Desconhecido")

    # Alternar o processamento dos quadros
    processar_este_quadro = not processar_este_quadro

    # Exibir os resultados
    for (top, right, bottom, left), nome in zip(localizacao_dos_rostos, nomes_detectados):
        # Redimensionar as coordenadas para o tamanho original do quadro
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Desenhar um retângulo ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Desenhar um retângulo com o nome embaixo do rosto
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, nome, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Exibir a imagem com a câmera
    cv2.imshow('Webcam_facerecognition', frame)

    # Pressionar 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e fechar as janelas
video_capture.release()
cv2.destroyAllWindows()
