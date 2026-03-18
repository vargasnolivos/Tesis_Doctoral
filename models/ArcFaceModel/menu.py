from ArcFaceModel import ArcFaceModel


def main():

    arcface = ArcFaceModel()

    while True:

        print("\n===== MENU ARCFACE =====")
        print("1. Entrenar modelo")
        print("2. Evaluar modelo")
        print("3. Reconocimiento en tiempo real")
        print("4. Salir")

        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            arcface.train_model()

        elif opcion == "2":
            arcface.evaluate_model()

        elif opcion == "3":
            arcface.test_model()

        elif opcion == "4":
            break

        else:
            print("Opción inválida")


if __name__ == "__main__":
    main()