from LBPH import LBPH


def main():

    lbph = LBPH()

    while True:

        print("\n====================================")
        print("   SISTEMA DE RECONOCIMIENTO FACIAL")
        print("====================================")
        print("1. Entrenar modelo")
        print("2. Evaluar modelo")
        print("3. Reconocimiento en tiempo real")
        print("4. Salir")
        print("====================================")

        opcion = input("Seleccione una opción: ")

        if opcion == "1":

            print("\nEntrenando modelo...\n")
            lbph.train_model()

        elif opcion == "2":

            print("\nEvaluando modelo...\n")
            lbph.evaluate_model()

        elif opcion == "3":

            print("\nIniciando reconocimiento facial...\n")
            lbph.test_model()

        elif opcion == "4":

            print("Saliendo del sistema...")
            break

        else:

            print("Opción no válida. Intente nuevamente.")


if __name__ == "__main__":
    main()