from ArcFaceModel import ArcFaceModel

def main():
    model = ArcFaceModel()

    while True:
        print("\n===== MENÚ ARCFACE =====")
        print("1. Generar embeddings")
        print("2. Evaluar KNN")
        print("3. Evaluar SVM")
        print("4. Evaluar Decision Tree")
        print("5. Evaluar Random Forest")
        print("6. Probar cámara")
        print("7. Salir")

        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            model.train_model()

        elif opcion == "2":
            if model.embeddings is None or model.labels is None:
                model.train_model()
            try:
                k = int(input("Ingrese el valor de k para KNN: "))
                model.evaluate(classifier_type="KNN", k=k)
            except ValueError:
                print("Por favor ingrese un número entero válido para k.")

        elif opcion == "3":
            if model.embeddings is None or model.labels is None:
                model.train_model()
            model.evaluate(classifier_type="SVM")

        elif opcion == "4":
            if model.embeddings is None or model.labels is None:
                model.train_model()
            model.evaluate(classifier_type="DecisionTree")

        elif opcion == "5":
            if model.embeddings is None or model.labels is None:
                model.train_model()
            model.evaluate(classifier_type="RandomForest")

        elif opcion == "6":
            if model.embeddings is None or model.labels is None:
                model.train_model()
            model.test_model()

        elif opcion == "7":
            print("Saliendo...")
            break

        else:
            print("Opción inválida, intente nuevamente.")

if __name__ == "__main__":
    main()