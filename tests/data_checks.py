from pathlib import Path

def check_data_structure():
    base_dir = Path(__file__).parent.parent / 'data' / 'data'

    required_dirs = [
        base_dir / 'libristutter' / 'episodes',
        base_dir / 'libristutter' / 'labels',
        base_dir / 'sep28k' / 'episodes',
        base_dir / 'sep28k' / 'labels'
    ]

    for dir_path in required_dirs:
        assert dir_path.exists(), f"Missing directory: {dir_path}"
        print(f" {dir_path.relative_to(base_dir.parent)}")

    print("All required directories exist")

if __name__ == "__main__":
    check_data_structure()
