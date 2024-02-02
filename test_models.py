# test_models.py

import multiprocessing
from model import create_model
from tensorflow.keras.datasets import mnist

def train_and_evaluate(i):
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

    # Create a model
    model = create_model()

    # Train the model
    model.fit(x_train, y_train, epochs=5, verbose=0)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Model {i}: Accuracy {accuracy}, Loss {loss}")

if __name__ == "__main__":
    processes = []
    for i in range(4):  
        p = multiprocessing.Process(target=train_and_evaluate, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
