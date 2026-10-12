import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path("data")

def main():
    DATA_DIR.mkdir(exist_ok=True)
    extracted = DATA_DIR / "ml-latest-small"
    if extracted.exists():
        print("Data already available in", extracted)
        return
    zip_path = DATA_DIR / "ml-latest-small.zip"
    if not zip_path.exists():
        print("Downloading MovieLens...")
        urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    print("Done. Files are in", extracted)

if __name__ == "__main__":
     main()