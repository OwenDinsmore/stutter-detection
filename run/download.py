import subprocess
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    scripts_dir = project_root / 'data' / 'download_scripts'
    scripts = ['libristutter.py', 'sep28k.py']
    for script in scripts:
        subprocess.run([sys.executable, str(scripts_dir / script)])
if __name__ == "__main__":
    main()