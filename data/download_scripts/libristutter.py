import os
import requests
import zipfile #zip converter 
import subprocess #rar converter (mp3/mp4)
from pathlib import Path

def download_libristutter():
    base_dir = Path(__file__).parent.parent / 'data' / 'data'
    libristutter_dir = base_dir / 'libristutter'
    libristutter_dir.mkdir(parents=True, exist_ok=True) 
    doi_url = "https://borealisdata.ca/api/access/dataset/:persistentId/?persistentId=doi:10.5683/SP3/NKVOGQ"
    
    archive_path = libristutter_dir / 'libristutter.zip'
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
                print(f"\rProgress: {progress:.1f}%", end='')   
    try:
        subprocess.run(['unrar', 'x', '-y', str(archive_path), str(libristutter_dir)], check=True)
    except subprocess.CalledProcessError:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(libristutter_dir)
    archive_path.unlink()
    print("done")

if __name__ == "__main__":
    download_libristutter()