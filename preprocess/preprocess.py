import pandas as pd
import numpy as np
from pathlib import Path
import librosa 

class SEP28kPreprocessor:
    def __init__(self, data_dir='data/data/sep28k', sr=16000):
        self.data_dir = Path(data_dir)
        self.sr = sr
        self.labels_df = pd.read_csv(self.data_dir / 'labels/SEP-28k_labels.csv', skipinitialspace=True)
        self.stutter_cols = ['Prolongation', 'Block', 'SoundRep', 'WordRep']

    def load_clip(self, row):
        ep_file = self.data_dir / f'episodes/episode_{row.EpId}.mp3'
        if not ep_file.exists():
            return None, None
        audio, _ = librosa.load(ep_file, sr=self.sr, offset=row.Start/self.sr, duration=(row.Stop-row.Start)/self.sr)
        label = 1 if row[self.stutter_cols].sum() > 0 else 0
        return audio, label

    def extract_features(self, audio):
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        return mfcc.mean(axis=1)

    def get_dataset(self, max_samples=None):
        X, y = [], []
        df = self.labels_df if max_samples is None else self.labels_df.head(max_samples)

        for _, row in df.iterrows():
            audio, label = self.load_clip(row)
            if audio is not None:
                features = self.extract_features(audio)
                X.append(features)
                y.append(label)

        return np.array(X), np.array(y)

if __name__ == "__main__":
    prep = SEP28kPreprocessor()
    X, y = prep.get_dataset(max_samples=100)
    print(f"features shape: {X.shape}")
    print(f"labels shape: {y.shape}")
    print(f"stutter samples: {y.sum()}, clean samples: {(y==0).sum()}")
