import os
import cv2
import numpy as np
import time

from tqdm import tqdm
from deepface import DeepFace

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ArcFaceModel:

    def __init__(self, dataset_path="dataset"):

        self.dataset_path = dataset_path
        self.embeddings = None
        self.labels = None


    # ======================================
    # ENTRENAMIENTO
    # ======================================
    def train_model(self):

        print("\nGenerando embeddings con ArcFace...\n")

        embeddings = []
        labels = []

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

        for img_path, person_name in tqdm(
            zip(image_paths, image_labels),
            total=len(image_paths),
            desc="Procesando imágenes",
            unit="img"
        ):

            try:

                result = DeepFace.represent(
                    img_path=img_path,
                    model_name="ArcFace",
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


    # ======================================
    # EVALUACIÓN
    # ======================================
    def evaluate_model(self):

        X_train, X_test, y_train, y_test = train_test_split(
            self.embeddings,
            self.labels,
            test_size=0.2,
            random_state=42
        )

        classifier = KNeighborsClassifier(n_neighbors=3)

        classifier.fit(X_train, y_train)

        start = time.time()

        predictions = classifier.predict(X_test)

        end = time.time()
        # tiempos
        total_time = end - start
        time_per_image = total_time / len(X_test)
        accuracy = accuracy_score(y_test, predictions)
        #Metricas
        precision = precision_score(y_test, predictions, average="weighted")
        recall = recall_score(y_test, predictions, average="weighted")
        f1 = f1_score(y_test, predictions, average="weighted")

        print("\n===== RESULTADOS ARCFACE =====")

        print("Accuracy :", round(accuracy,4))
        print("Precision:", round(precision,4))
        print("Recall   :", round(recall,4))
        print("F1-score :", round(f1,4))

        print("Tiempo:", round(total_time,4))
        print("Tiempo por imagen:", round(time_per_image,4))


    # ======================================
    # PRUEBA EN TIEMPO REAL
    # ======================================
    def test_model(self):

        camera = cv2.VideoCapture(0)

        print("Reconocimiento facial ArcFace iniciado")

        while True:

            ret, frame = camera.read()

            if not ret:
                break

            try:

                result = DeepFace.find(
                    img_path=frame,
                    db_path=self.dataset_path,
                    model_name="ArcFace",
                    enforce_detection=False
                )

                print(result)

            except:
                pass

            cv2.imshow("ArcFace Recognition", frame)

            if cv2.waitKey(1) == 27:
                break

        camera.release()
        cv2.destroyAllWindows()