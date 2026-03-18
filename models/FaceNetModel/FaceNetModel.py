import os
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from tqdm import tqdm


class FaceNetModel:

    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path

    # ======================================
    # ENTRENAR (generar embeddings)
    # ======================================
    def train_model(self):
        print("Generando embeddings con FaceNet...")

        embeddings = []
        labels = []

        # contar total de imágenes
        image_paths = []
        image_labels = []

        for person_name in os.listdir(self.dataset_path):

            person_path = os.path.join(self.dataset_path, person_name)

            if not os.path.isdir(person_path):
                continue

            for img_name in os.listdir(person_path):

                img_path = os.path.join(person_path, img_name)

                image_paths.append(img_path)
                image_labels.append(person_name)

        print("Total de imágenes:", len(image_paths))

        start = time.time()

        # barra de progreso
        for img_path, person_name in tqdm(
            zip(image_paths, image_labels),
            total=len(image_paths),
            desc="Procesando imágenes",
            unit="img"
        ):

            try:

                result = DeepFace.represent(
                    img_path=img_path,
                    model_name="Facenet",
                    enforce_detection=False
                )

                embedding = result[0]["embedding"]

                embeddings.append(embedding)
                labels.append(person_name)

            except:
                continue

        end = time.time()

        self.embeddings = np.array(embeddings)
        self.labels = np.array(labels)

        print("\nEmbeddings generados:", len(self.embeddings))
        print("Tiempo total:", round((end-start)/60,2), "minutos")
        np.save("facenet_embeddings.npy", self.embeddings)
        np.save("facenet_labels.npy", self.labels)
        print("Embeddings guardados en disco")

    # ======================================
    # EVALUAR
    # ======================================
    def evaluate_model(self):

        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier

        X_train, X_test, y_train, y_test = train_test_split(
            self.embeddings,
            self.labels,
            test_size=0.2,
            random_state=42
        )

        classifier = KNeighborsClassifier(n_neighbors=1)

        classifier.fit(X_train, y_train)

        start = time.time()

        predictions = classifier.predict(X_test)

        end = time.time()
        total_time = end - start
        time_per_image = total_time / len(X_test)

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(y_test, predictions, average="weighted")

        recall = recall_score(y_test, predictions, average="weighted")

        f1 = f1_score(y_test, predictions, average="weighted")

        print("\n===== RESULTADOS FACENET =====")

        print("Accuracy :", round(accuracy,4))
        print("Precision:", round(precision,4))
        print("Recall   :", round(recall,4))
        print("F1-score :", round(f1,4))
        print("Tiempo Total:", round(total_time,6))
        print("Tiempo por imagen:", round(time_per_image,6), "segundos")

    # ======================================
    # TEST CON CAMARA
    # ======================================

    def test_model(self):

        import time
        import numpy as np
        import cv2
        from deepface import DeepFace

        print("Iniciando reconocimiento facial FaceNet...")

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            print("Error: No se pudo abrir la cámara")
            return

        while True:

            ret, frame = camera.read()

            if not ret:
                break

            start = time.time()

            try:

                # generar embedding del rostro detectado
                result = DeepFace.represent(
                    img_path=frame,
                    model_name="Facenet",
                    enforce_detection=False
                )

                embedding = np.array(result[0]["embedding"])

                # calcular distancia contra todos los embeddings
                distances = np.linalg.norm(self.embeddings - embedding, axis=1)

                min_index = np.argmin(distances)

                name = self.labels[min_index]

                distance = distances[min_index]

            except:

                name = "Desconocido"
                distance = 0

            end = time.time()

            recognition_time = round(end - start, 3)

            # mostrar texto en pantalla
            cv2.putText(
                frame,
                f"Persona: {name}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Distancia: {distance:.3f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Tiempo: {recognition_time}s",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow("Reconocimiento Facial FaceNet", frame)

            # ESC para salir
            if cv2.waitKey(1) & 0xFF == 27:
                break

        camera.release()
        cv2.destroyAllWindows()