# evaluate.py

from preprocessing import load_and_preprocess_data
from model import create_model
import matplotlib.pyplot as plt

def evaluate_model():
    _, _, x_test, y_test = load_and_preprocess_data()
    model = create_model()
    model.load_weights('path_to_model_weights.h5')  # Load your trained model weights

    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Optional: Add code to visualize test results, confusion matrix, etc.

if __name__ == "__main__":
    evaluate_model()
