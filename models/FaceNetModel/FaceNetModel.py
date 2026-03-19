import os
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time
from tqdm import tqdm


class FaceNetModel:

    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.embeddings = None
        self.labels = None

    # ======================================
    # GENERAR EMBEDDINGS
    # ======================================
    def train_model(self):
        print("Generando embeddings con FaceNet...")

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

        for img_path, person_name in tqdm(zip(image_paths, image_labels), total=len(image_paths), desc="Procesando imágenes", unit="img"):
            try:
                result = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)
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
        print("✅ Embeddings guardados en disco")

    # ======================================
    # CARGAR EMBEDDINGS
    # ======================================
    def load_embeddings(self):
        if os.path.exists("facenet_embeddings.npy") and os.path.exists("facenet_labels.npy"):
            self.embeddings = np.load("facenet_embeddings.npy")
            self.labels = np.load("facenet_labels.npy")
            print("✅ Embeddings cargados desde disco")
            return True
        else:
            print("❌ No existen embeddings guardados")
            return False

    # ======================================
    # NORMALIZAR EMBEDDINGS
    # ======================================
    def normalize_embeddings(self):
        self.embeddings = Normalizer().fit_transform(self.embeddings)

    # ======================================
    # EVALUAR CUALQUIER CLASIFICADOR
    # ======================================
    def evaluate(self, classifier_type="KNN", k=3):
        self.normalize_embeddings()
        X_train, X_test, y_train, y_test = train_test_split(self.embeddings, self.labels, test_size=0.2, random_state=42)

        # Selección del clasificador
        if classifier_type == "KNN":
            classifier = KNeighborsClassifier(n_neighbors=k)
        elif classifier_type == "SVM":
            classifier = SVC(kernel='linear')
        elif classifier_type == "DecisionTree":
            classifier = DecisionTreeClassifier()
        elif classifier_type == "RandomForest":
            classifier = RandomForestClassifier(n_estimators=100)
        else:
            raise ValueError("Clasificador no soportado")

        classifier.fit(X_train, y_train)

        start = time.time()
        predictions = classifier.predict(X_test)
        end = time.time()

        total_time = end - start
        time_per_image = total_time / len(X_test)

        print(f"\n===== FACENET + {classifier_type} =====")
        if classifier_type=="KNN":
            print(f"k = {k}")
        print("Accuracy :", round(accuracy_score(y_test, predictions),4))
        print("Precision:", round(precision_score(y_test, predictions, average="weighted"),4))
        print("Recall   :", round(recall_score(y_test, predictions, average="weighted"),4))
        print("F1-score :", round(f1_score(y_test, predictions, average="weighted"),4))
        print("Tiempo Total:", round(total_time,6), "s")
        print("Tiempo por imagen:", round(time_per_image,6), "s")

    # ======================================
    # TEST CON CAMARA
    # ======================================
    def test_model(self):
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
                result = DeepFace.represent(img_path=frame, model_name="Facenet", enforce_detection=False)
                embedding = np.array(result[0]["embedding"])
                distances = np.linalg.norm(self.embeddings - embedding, axis=1)
                min_index = np.argmin(distances)
                name = self.labels[min_index]
                distance = distances[min_index]
            except:
                name = "Desconocido"
                distance = 0
            end = time.time()
            recognition_time = round(end - start, 3)
            cv2.putText(frame, f"Persona: {name}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Distancia: {distance:.3f}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Tiempo: {recognition_time}s", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("FaceNet", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        camera.release()
        cv2.destroyAllWindows()