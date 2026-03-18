import os
import cv2
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical


class CNN:

    def __init__(self, dataset_path="dataset", model_path="models/cnn_model.h5"):

        self.dataset_path = dataset_path
        self.model_path = model_path

        os.makedirs("models", exist_ok=True)

    # ======================================
    # CARGAR DATASET
    # ======================================

    def load_dataset(self):

        faces = []
        labels = []
        label_map = {}

        label_id = 0

        for person_name in os.listdir(self.dataset_path):

            person_path = os.path.join(self.dataset_path, person_name)

            if not os.path.isdir(person_path):
                continue

            label_map[label_id] = person_name

            for img_name in os.listdir(person_path):

                img_path = os.path.join(person_path, img_name)

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                img = cv2.resize(img, (100, 100))

                faces.append(img)
                labels.append(label_id)

            label_id += 1

        faces = np.array(faces) / 255.0
        faces = faces.reshape(-1, 100, 100, 1)

        labels = to_categorical(labels)

        return faces, labels, label_map

    # ======================================
    # ENTRENAR MODELO
    # ======================================

    def train_model(self):

        X, y, label_map = self.load_dataset()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = Sequential()

        model.add(Conv2D(32, (3,3), activation='relu', input_shape=(100,100,1)))
        model.add(MaxPooling2D((2,2)))

        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(MaxPooling2D((2,2)))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))

        model.add(Dense(y.shape[1], activation='softmax'))

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print("Entrenando CNN...")

        model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

        model.save(self.model_path)

        print("Modelo CNN guardado en:", self.model_path)

    # ======================================
    # EVALUAR MODELO
    # ======================================

    def evaluate_model(self):

        X, y, label_map = self.load_dataset()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = load_model(self.model_path)

        start_time = time.time()

        predictions = model.predict(X_test)

        end_time = time.time()

        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)

        accuracy = accuracy_score(y_true, y_pred)

        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")

        total_time = end_time - start_time
        time_per_image = total_time / len(X_test)

        print("\n===== RESULTADOS CNN =====")

        print("Accuracy :", round(accuracy,4))
        print("Precision:", round(precision,4))
        print("Recall   :", round(recall,4))
        print("F1-score :", round(f1,4))
        print("Tiempo total:", round(total_time,4))
        print("Tiempo por imagen:", round(time_per_image,4))

    # ======================================
    # RECONOCIMIENTO EN TIEMPO REAL
    # ======================================

    def test_model(self):

        model = load_model(self.model_path)

        label_map = {}
        label_id = 0

        for person_name in os.listdir(self.dataset_path):

            person_path = os.path.join(self.dataset_path, person_name)

            if not os.path.isdir(person_path):
                continue

            label_map[label_id] = person_name
            label_id += 1

        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        camera = cv2.VideoCapture(0)

        print("Reconocimiento facial CNN iniciado")

        while True:

            ret, frame = camera.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces:

                face = gray[y:y+h, x:x+w]

                face = cv2.resize(face, (100,100))

                face = face / 255.0

                face = face.reshape(1,100,100,1)

                prediction = model.predict(face)

                label = np.argmax(prediction)

                name = label_map.get(label, "Desconocido")

                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                cv2.putText(
                    frame,
                    name,
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

            cv2.imshow("CNN Face Recognition", frame)

            if cv2.waitKey(1) == 27:
                break

        camera.release()
        cv2.destroyAllWindows()