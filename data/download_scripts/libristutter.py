import requests
import zipfile
import subprocess
import shutil
from pathlib import Path

def download_libristutter():
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / 'raw' / 'libristutter'
    processed_dir = base_dir / 'data' / 'libristutter'

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_episodes = processed_dir / 'episodes'
    processed_labels = processed_dir / 'labels'
    processed_episodes.mkdir(parents=True, exist_ok=True)
    processed_labels.mkdir(parents=True, exist_ok=True)

    doi_url = "https://borealisdata.ca/api/access/dataset/:persistentId/?persistentId=doi:10.5683/SP3/NKVOGQ"
    archive_path = raw_dir / 'libristutter.zip'

    if not archive_path.exists():
        response = requests.get(doi_url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(archive_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\rprogress: {progress:.1f}%", end='')

    try:
        subprocess.run(['unrar', 'x', '-y', str(archive_path), str(raw_dir)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)

    for item in raw_dir.iterdir():
        if item.is_file() and item.suffix in ['.wav', '.mp3', '.flac']:
            shutil.copy2(item, processed_episodes / item.name)
        elif item.is_file() and item.suffix in ['.csv', '.txt', '.json']:
            shutil.copy2(item, processed_labels / item.name)

    print("\ndone")

if __name__ == "__main__":
    download_libristutter()
