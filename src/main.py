# src/main.py

import cv2
from object_detector import ObjectDetector

def main():
    print("Inicializando detector YOLOv5...")
    detector = ObjectDetector()

    print("Abrindo webcam (backend DirectShow)...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # usa DirectShow no Windows
    if not cap.isOpened():
        print("Erro: não foi possível abrir a webcam com índice 0.")
        return

    print("Webcam aberta com sucesso! Entrando no loop de captura.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Aviso: falha ao capturar frame. Tentando novamente...")
            continue

        # detecção
        detections = detector.detect(frame)

        # desenha cada detecção
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = det['class']
            conf  = det['confidence']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # mostra resultado
        cv2.imshow("Real-time Object Detection", frame)

        # sai ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Saiu pelo usuário.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Finalizado.")

if __name__ == "__main__":
    main()
