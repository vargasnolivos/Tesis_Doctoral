from FaceNetModel import FaceNetModel

def main():
    model = FaceNetModel()

    while True:
        print("\n===== MENÚ FACENET =====")
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
            if model.load_embeddings():
                try:
                    k = int(input("Ingrese el valor de k para KNN: "))
                    if k < 1:
                        print("❌ k debe ser mayor o igual a 1")
                        continue
                except ValueError:
                    print("❌ Debe ingresar un número entero")
                    continue
                model.evaluate(classifier_type="KNN", k=k)

        elif opcion == "3":
            if model.load_embeddings():
                model.evaluate(classifier_type="SVM")

        elif opcion == "4":
            if model.load_embeddings():
                model.evaluate(classifier_type="DecisionTree")

        elif opcion == "5":
            if model.load_embeddings():
                model.evaluate(classifier_type="RandomForest")

        elif opcion == "6":
            if model.load_embeddings():
                model.test_model()

        elif opcion == "7":
            print("Saliendo...")
            break

        else:
            print("Opción inválida")

if __name__ == "__main__":
    main()