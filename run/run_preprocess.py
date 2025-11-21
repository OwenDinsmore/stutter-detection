import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.download_scripts.sep28k import download_sep28k
from data.download_scripts.libristutter import download_libristutter
from data.download_scripts.extract_clips import extract_clips

def main():
    base = Path(__file__).parent.parent / 'data'

    print("=== Downloading LibriStutter ===")
    download_libristutter()

    print("=== Downloading SEP-28k ===")
    download_sep28k()

    print("=== Extracting SEP-28k clips ===")
    labels = base / 'data' / 'sep28k' / 'labels' / 'SEP-28k_labels.csv'
    wavs = base / 'data' / 'sep28k' / 'episodes'
    clips = base / 'data' / 'sep28k' / 'clips'
    if wavs.exists() and any(wavs.glob('*.wav')):
        extract_clips(str(labels), str(wavs), str(clips), progress=True)
    else:
        print("No WAV files found, skipping clip extraction")

    print("=== Done ===")

if __name__ == "__main__":
    main()