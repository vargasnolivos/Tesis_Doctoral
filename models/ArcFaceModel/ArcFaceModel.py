import os
import cv2
import numpy as np
import time
from tqdm import tqdm
from deepface import DeepFace
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer


class ArcFaceModel:

    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.embeddings = None
        self.labels = None

    # ======================================
    # GENERAR O CARGAR EMBEDDINGS
    # ======================================
    def train_model(self, force_retrain=False):
        if not force_retrain and os.path.exists("arcface_embeddings.npy") and os.path.exists("arcface_labels.npy"):
            print("Cargando embeddings desde disco...")
            self.embeddings = np.load("arcface_embeddings.npy")
            self.labels = np.load("arcface_labels.npy")
            print("Embeddings cargados:", len(self.embeddings))
            return

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

        # Guardar embeddings
        np.save("arcface_embeddings.npy", self.embeddings)
        np.save("arcface_labels.npy", self.labels)

        print("\nEmbeddings generados:", len(self.embeddings))
        print("Tiempo total:", round((end-start)/60,2), "minutos")
        print("Embeddings guardados en disco")

    # ======================================
    # NORMALIZAR EMBEDDINGS
    # ======================================
    def normalize_embeddings(self):
        if self.embeddings is not None:
            self.embeddings = Normalizer().fit_transform(self.embeddings)

    # ======================================
    # EVALUAR MODELO
    # ======================================
    def evaluate(self, classifier_type="KNN", k=3):
        if self.embeddings is None or self.labels is None:
            print("Primero debes generar los embeddings")
            return

        self.normalize_embeddings()

        X_train, X_test, y_train, y_test = train_test_split(
            self.embeddings,
            self.labels,
            test_size=0.2,
            random_state=42
        )

        if classifier_type == "KNN":
            classifier = KNeighborsClassifier(n_neighbors=k)
        elif classifier_type == "SVM":
            classifier = SVC(kernel="linear")
        elif classifier_type == "DecisionTree":
            classifier = DecisionTreeClassifier()
        elif classifier_type == "RandomForest":
            classifier = RandomForestClassifier(n_estimators=100)
        else:
            raise ValueError("Clasificador no reconocido")

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

        print(f"\n===== RESULTADOS ARCFACE ({classifier_type}) =====")
        if classifier_type == "KNN":
            print(f"KNN (k={k})")
        print("Accuracy :", round(accuracy,4))
        print("Precision:", round(precision,4))
        print("Recall   :", round(recall,4))
        print("F1-score :", round(f1,4))
        print("Tiempo total:", round(total_time,6), "s")
        print("Tiempo por imagen:", round(time_per_image,6), "s")

    # ======================================
    # PRUEBA EN TIEMPO REAL
    # ======================================
    def test_model(self):
        if self.embeddings is None or self.labels is None:
            print("Primero debes generar los embeddings")
            return

        camera = cv2.VideoCapture(0)
        print("Reconocimiento facial ArcFace iniciado")

        while True:
            ret, frame = camera.read()
            if not ret:
                break

            start = time.time()
            try:
                result = DeepFace.represent(
                    img_path=frame,
                    model_name="ArcFace",
                    enforce_detection=False
                )
                embedding = np.array(result[0]["embedding"])
                distances = np.linalg.norm(self.embeddings - embedding, axis=1)
                min_index = np.argmin(distances)
                name = self.labels[min_index]
                distance = distances[min_index]

            except:
                name = "Desconocido"
                distance = 0

            end = time.time()
            recognition_time = round(end - start, 4)

            cv2.putText(frame, f"Persona: {name}", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Distancia: {distance:.3f}", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(frame, f"Tiempo: {recognition_time}s", (30,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.imshow("ArcFace Recognition", frame)
            if cv2.waitKey(1) == 27:  # ESC
                break

        camera.release()
        cv2.destroyAllWindows()