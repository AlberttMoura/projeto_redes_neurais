import sys
from tensorflow import keras


def main():
    model_type = int(input(
        "Selecione o modelo da rede neural: [MLP(1), Convolucional(2)]: "))

    # Extração e separação dos dados de treino e teste da base MNIST
    (train_images, train_labels), (test_images,
                                   test_labels) = keras.datasets.mnist.load_data()

    # Normaliza os pixels para valores até 255
    train_images = train_images.astype("float32") / 255.0
    # Normaliza os pixels para valores até 255
    test_images = test_images.astype("float32") / 255.0

    # Define o modelo
    if model_type == 1:
        print("Utilizando modelo MLP")

        # Pré-processamento dos dados
        train_images = train_images.reshape((60000, 28 * 28))
        test_images = test_images.reshape((10000, 28 * 28))

        # Modelo MLP
        model = keras.Sequential([
            keras.layers.Dense(512, activation="relu",
                               input_shape=(28 * 28,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10, activation="softmax")
        ])

    elif model_type == 2:
        print("Utilizando modelo convolucional")

        # Pré-processamento dos dados
        train_images = train_images.reshape(
            (60000, 28, 28, 1))  # Add a channel dimension
        # Redimenciona os dados de teste para
        test_images = test_images.reshape((10000, 28, 28, 1))

        # Modelo Convolucional
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation="relu",
                                input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10, activation="softmax")
        ])
    else:
        print("Nenhum modelo válido selecionado")
        return
    # Compila o modelo
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Treina o modelo
    model.fit(train_images, train_labels, epochs=5)

    # Avalia o modelo
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Test accuracy:", test_acc)


if __name__ == "__main__":
    main()
