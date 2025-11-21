import subprocess
import requests
from pathlib import Path
import numpy as np

def download_sep28k():
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / 'raw' / 'sep28k'
    processed_dir = base_dir / 'data' / 'sep28k'

    raw_episodes = raw_dir / 'episodes'
    raw_labels = raw_dir / 'labels'
    processed_episodes = processed_dir / 'episodes'
    processed_labels = processed_dir / 'labels'

    for d in [raw_episodes, raw_labels, processed_episodes, processed_labels]:
        d.mkdir(parents=True, exist_ok=True)

    repo_base = "https://raw.githubusercontent.com/apple/ml-stuttering-events-dataset/main"
    csv_files = ['SEP-28k_episodes.csv', 'SEP-28k_labels.csv']

    for file in csv_files:
        url = f"{repo_base}/{file}"
        response = requests.get(url)
        response.raise_for_status()
        (raw_labels / file).write_bytes(response.content)
        (processed_labels / file).write_bytes(response.content)

    episodes_file = raw_labels / 'SEP-28k_episodes.csv'
    table = np.loadtxt(episodes_file, dtype=str, delimiter=',')
    urls = table[:, 2]
    audio_types = ['.mp3', '.m4a', '.mp4']

    print(f"downloading {len(urls)} files")

    for i, url in enumerate(urls):
        try:
            audio_ext = next((ext for ext in audio_types if ext in url), '.mp3')
            raw_file = raw_episodes / f"episode_{i}{audio_ext}"
            wav_file = processed_episodes / f"episode_{i}.wav"

            if wav_file.exists():
                continue

            print(f"\rdownloading {i+1}/{len(urls)}", end='')

            if not raw_file.exists():
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                raw_file.write_bytes(response.content)

            subprocess.run([
                'ffmpeg', '-i', str(raw_file),
                '-ar', '16000', '-ac', '1',
                str(wav_file), '-y'
            ], capture_output=True, check=True)

        except Exception as e:
            print(f"\nerror on {url}: {str(e)}")
            continue

    print("\nSEP-28k download complete")

if __name__ == "__main__":
    download_sep28k()
