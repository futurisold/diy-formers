import os
import requests
from pathlib import Path


DIRECTORY = Path('/tmp/wmt14-un-corpus')
FILENAME  = 'wmt14-un-corpus.tgz'
URL       = 'https://www.statmt.org/wmt13/training-parallel-un.tgz'


def download_file():
    if not os.path.exists(DIRECTORY / FILENAME):
        print(f'Downloading {FILENAME}...')
        response = requests.get(URL)
        os.makedirs(DIRECTORY, exist_ok=True)
        with open(DIRECTORY / FILENAME, 'wb') as f: f.write(response.content)
    else: print(f'File {FILENAME} already exists.')


def extract_file():
    if os.path.exists(DIRECTORY / FILENAME):
        print(f'Extracting {FILENAME}...')
        os.system(f'tar -xzf {DIRECTORY / FILENAME} -C {DIRECTORY}')
    else: print(f'File {FILENAME} does not exist.')


if __name__ == '__main__':
    download_file()
    extract_file()

