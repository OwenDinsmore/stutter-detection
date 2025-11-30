import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers

def build_cnn(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], 1))(inputs)

    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def cnn_model(samples_num=100, test_size=0.3, epochs=50, batch_size=32):
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from preprocess.preprocess import SEP28kPreprocessor
    print(f"Loading SEP28k data, max sample size: {samples_num}")
    prep = SEP28kPreprocessor()
    X, y = prep.get_dataset(max_samples=samples_num)
    print(f"Data loaded")
    print(f"features shape: {X.shape}")
    print(f"labels shape: {y.shape}")
    print(f"stutter samples: {y.sum()}, clean samples: {(y==0).sum()}")

    print(f"Training CNN Model, test size: {test_size}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    model = build_cnn(input_shape=(X_train.shape[1],))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"done. accuracy: {accuracy}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        samples_num = int(sys.argv[1])
        test_size = float(sys.argv[2])
        cnn_model(samples_num, test_size)
    elif len(sys.argv) == 2:
        samples_num = int(sys.argv[1])
        cnn_model(samples_num)
    else:
        cnn_model()
