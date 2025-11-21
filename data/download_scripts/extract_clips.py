import argparse
from pathlib import Path
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm
def extract_clips(labels_path, wavs_dir, clips_dir, progress=False):
    data = pd.read_csv(labels_path, dtype={"EpId": str})
    clips_dir = Path(clips_dir)
    wavs_dir = Path(wavs_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)

    loaded_wav = ""
    audio = None
    sample_rate = None

    items = range(len(data))
    if progress:
        items = tqdm(items)

    for i in items:
        row = data.iloc[i]
        show = row.Show
        episode = str(row.EpId).strip()
        clip_idx = row.ClipId
        start = row.Start
        stop = row.Stop

        wav_path = wavs_dir / f"{show}" / f"{episode}.wav"
        if not wav_path.exists():
            wav_path = wavs_dir / f"episode_{episode}.wav"
            if not wav_path.exists():
                continue

        clip_subdir = clips_dir / show / episode
        clip_subdir.mkdir(parents=True, exist_ok=True)
        clip_path = clip_subdir / f"{show}_{episode}_{clip_idx}.wav"

        if str(wav_path) != loaded_wav:
            sample_rate, audio = wavfile.read(wav_path)
            loaded_wav = str(wav_path)

        clip = audio[start:stop]
        wavfile.write(clip_path, sample_rate, clip)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, required=True)
    parser.add_argument('--wavs', type=str, default="wavs")
    parser.add_argument('--clips', type=str, default="clips")
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()
    extract_clips(args.labels, args.wavs, args.clips, args.progress)

if __name__ == "__main__":
    main()
