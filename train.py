# train.py

from preprocessing import load_and_preprocess_data
from model import create_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_model():
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    model = create_model()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)

    history = model.fit(x_train, y_train, 
                        epochs=50, 
                        batch_size=64, 
                        validation_data=(x_test, y_test),
                        callbacks=[early_stopping, reduce_lr])

    return history

if __name__ == "__main__":
    train_model()
