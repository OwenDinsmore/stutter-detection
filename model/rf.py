# random forests approach

import pandas as pd
import numpy as np
from pathlib import Path
import librosa
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def rf_model(samples_num=100, test_size=0.3):
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from preprocess.preprocess import SEP28kPreprocessor
    print(f"=== Loading SEP28k data, max sample size: {samples_num} ===")
    prep = SEP28kPreprocessor()
    X, y = prep.get_dataset(max_samples=samples_num)
    print(f"=== Data loaded ===")
    print(f"features shape: {X.shape}")
    print(f"labels shape: {y.shape}")
    print(f"stutter samples: {y.sum()}, clean samples: {(y==0).sum()}")

    print(f"=== Training Random Forests Model, test size: {test_size} ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    rf = RandomForestClassifier()
    params = {
    'n_estimators': [100,500,700],
    'max_depth': [1,5,7,None],
    'min_samples_split': [2,4,8]
    }
    gridsearch = GridSearchCV(rf, param_grid=params)
    gridsearch.fit(X_train,y_train)
    accuracy = gridsearch.score(X_test,y_test)
    print(f"=== Params found by Gridsearch: {gridsearch.best_params_} ===")
    print(f"=== Model Complete. Accuracy: {accuracy} ===")

if __name__ == "__main__":
    if len(sys.argv) == 3:   
        samples_num = int(sys.argv[1])
        test_size = float(sys.argv[2])
        rf_model(samples_num, test_size)
    elif len(sys.argv) == 2:
        samples_num = int(sys.argv[1])
        rf_model(samples_num)
    else:
        rf_model()