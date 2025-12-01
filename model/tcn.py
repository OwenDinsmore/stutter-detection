import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers

sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocess.data_augmentation import augment_mfcc

def build_tcn(input_shape, num_filters=64, kernel_size=3, dilations=[1, 2, 4, 8, 16, 32]):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for dilation in dilations:
        x = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            padding='causal'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def tcn_model(samples_num=100, test_size=0.3, epochs=50, batch_size=32, augment=False, augment_factor=2):
    from preprocess.preprocess import SEP28kPreprocessor
    print(f"Loading SEP28k data, max sample size: {samples_num}")
    prep = SEP28kPreprocessor()
    X, y = prep.get_dataset(max_samples=samples_num, temporal=True)
    print(f"Data loaded")
    print(f"features shape: {X.shape}")
    print(f"labels shape: {y.shape}")
    print(f"stutter samples: {y.sum()}, clean samples: {(y==0).sum()}")

    print(f"Training TCN Model, test size: {test_size}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if augment:
        print(f"applying data augmentation, augment factor: {augment_factor}")
        X_aug_list = [X_train]
        y_aug_list = [y_train]

        for _ in range(augment_factor - 1):
            X_aug = np.array([augment_mfcc(x) for x in X_train])
            X_aug_list.append(X_aug)
            y_aug_list.append(y_train)

        X_train = np.concatenate(X_aug_list)
        y_train = np.concatenate(y_aug_list)
        print(f"augmented training data shape: {X_train.shape}")

    model = build_tcn(input_shape=(X_train.shape[1], X_train.shape[2]))
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
        tcn_model(samples_num, test_size)
    elif len(sys.argv) == 2:
        samples_num = int(sys.argv[1])
        tcn_model(samples_num)
    else:
        tcn_model()
