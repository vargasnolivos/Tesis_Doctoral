import cv2
import os
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


class LBPH:

    def __init__(self, dataset_path="dataset", model_path="models/lbph_model.xml"):
        self.dataset_path = dataset_path
        self.model_path = model_path
        os.makedirs("models", exist_ok=True)

    # ======================================
    # ENTRENAMIENTO DEL MODELO
    # ======================================
    def train_model(self):

        faces = []
        labels = []
        label_map = {}
        label_id = 0

        print("Cargando imágenes del dataset...")

        for person_name in os.listdir(self.dataset_path):

            person_path = os.path.join(self.dataset_path, person_name)

            if not os.path.isdir(person_path):
                continue

            print("Procesando:", person_name)
            label_map[label_id] = person_name

            for image_name in os.listdir(person_path):

                image_path = os.path.join(person_path, image_name)

                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print("Imagen ignorada:", image_path)
                    continue

                img = cv2.resize(img, (200, 200))

                faces.append(img)
                labels.append(label_id)

            label_id += 1

        faces = np.array(faces)
        labels = np.array(labels)

        print(f"Total de imágenes cargadas: {len(faces)}")

        print("Entrenando modelo LBPH...")

        recognizer = cv2.face.LBPHFaceRecognizer_create()

        recognizer.train(faces, labels)

        recognizer.write(self.model_path)

        print("=================================")
        print("Modelo LBPH entrenado correctamente")
        print("Modelo guardado en:", self.model_path)
        print("=================================")

    # ======================================
    # RECONOCIMIENTO EN TIEMPO REAL
    # ======================================
    def test_model(self):

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(self.model_path)

        label_map = {}
        label_id = 0

        for person_name in os.listdir(self.dataset_path):

            person_path = os.path.join(self.dataset_path, person_name)

            if not os.path.isdir(person_path):
                continue

            label_map[label_id] = person_name
            label_id += 1

        print("Sujetos cargados:", label_map)

        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        camera = cv2.VideoCapture(0)

        print("Sistema de reconocimiento facial iniciado...")
        print("Presione ESC para salir")

        while True:

            ret, frame = camera.read()

            if not ret:
                print("Error leyendo cámara")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(50, 50)
            )

            for (x, y, w, h) in faces:

                face = gray[y:y+h, x:x+w]

                try:

                    face = cv2.resize(face, (200, 200))

                    label, confidence = recognizer.predict(face)

                    name = label_map.get(label, "Desconocido")

                    text = f"{name} | Confianza: {round(confidence,2)}"

                except:
                    text = "Error reconocimiento"

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

                cv2.putText(
                    frame,
                    text,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

            cv2.imshow("Reconocimiento Facial - PRODOTICS", frame)

            key = cv2.waitKey(1)

            if key == 27:
                break

        camera.release()
        cv2.destroyAllWindows()

    # ======================================
    # EVALUACIÓN DEL MODELO
    # ======================================
    def evaluate_model(self):

        faces = []
        labels = []
        label_map = {}

        label_id = 0

        for person_name in os.listdir(self.dataset_path):

            person_path = os.path.join(self.dataset_path, person_name)

            if not os.path.isdir(person_path):
                continue

            images = []

            for image_name in os.listdir(person_path):

                img_path = os.path.join(person_path, image_name)

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                img = cv2.resize(img, (200, 200))

                images.append(img)

            if len(images) == 0:
                continue

            label_map[label_id] = person_name

            train_imgs, test_imgs = train_test_split(
                images,
                test_size=0.2,
                random_state=42
            )

            for img in train_imgs:
                faces.append(img)
                labels.append(label_id)

            if 'test_set' not in locals():
                test_set = []
                test_labels = []

            for img in test_imgs:
                test_set.append(img)
                test_labels.append(label_id)

            label_id += 1

        faces = np.array(faces)
        labels = np.array(labels)

        recognizer = cv2.face.LBPHFaceRecognizer_create()

        recognizer.train(faces, labels)

        recognizer.write(self.model_path)

        print("Modelo LBPH entrenado y guardado correctamente.")

        true_labels = test_labels
        predicted_labels = []

        start_time = time.time()

        for img in test_set:

            predicted_label, confidence = recognizer.predict(img)

            predicted_labels.append(predicted_label)

        end_time = time.time()

        total_time = end_time - start_time

        time_per_image = total_time / len(test_set)

        accuracy = accuracy_score(true_labels, predicted_labels)

        precision = precision_score(true_labels, predicted_labels, average="weighted")

        recall = recall_score(true_labels, predicted_labels, average="weighted")

        f1 = f1_score(true_labels, predicted_labels, average="weighted")

        print("\n===== RESULTADOS DEL MODELO =====")

        print(f"Accuracy  : {round(accuracy,4)}")
        print(f"Precision : {round(precision,4)}")
        print(f"Recall    : {round(recall,4)}")
        print(f"F1-score  : {round(f1,4)}")

        print(f"Tiempo total: {total_time:.2f} s")

        print(f"Tiempo por imagen: {time_per_image:.4f} s")