import csv
import subprocess
import requests
from pathlib import Path
import numpy as np

def download_sep28k():
    base_dir = Path(__file__).parent.parent / 'data' / 'data'
    sep28k_dir = base_dir / 'sep28k'
    episodes_dir = sep28k_dir / 'episodes'
    labels_dir = sep28k_dir / 'labels'
    episodes_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    repo_base = "https://raw.githubusercontent.com/apple/ml-stuttering-events-dataset/main"
    files_to_download = [
        'SEP-28k_episodes.csv',
        'SEP-28k_labels.csv',
        'download_audio.py',
        'extract_clips.py'
    ]

    for file in files_to_download:
        url = f"{repo_base}/{file}"
        response = requests.get(url)
        response.raise_for_status()
        if file.endswith('.csv'):
            (labels_dir / file).write_bytes(response.content)
        else:
            (sep28k_dir / file).write_bytes(response.content)

    episodes_file = labels_dir / 'SEP-28k_episodes.csv'
    labels_file = labels_dir / 'SEP-28k_labels.csv'
    table = np.loadtxt(episodes_file, dtype=str, delimiter=',')
    urls = table[:, 2]
    audio_types = ['.mp3', '.m4a', '.mp4']

    print(f"downloading {len(urls)} files")

    for i, url in enumerate(urls):
        try:
            audio_ext = next((ext for ext in audio_types if ext in url), '.mp3')
            output_file = episodes_dir / f"episode_{i}{audio_ext}"
            wav_file = episodes_dir / f"episode_{i}.wav"

            if wav_file.exists():
                continue

            print(f"\rdownloading {i+1}/{len(urls)}", end='')
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            output_file.write_bytes(response.content)

            subprocess.run([
                'ffmpeg', '-i', str(output_file),
                '-ar', '16000', '-ac', '1',
                str(wav_file), '-y'
            ], capture_output=True, check=True)
            output_file.unlink()

        except Exception as e:
            print(f"\nerror on {url}: {str(e)}")
            continue

    print("\nSEP-28k download complete")

if __name__ == "__main__":
    download_sep28k()
