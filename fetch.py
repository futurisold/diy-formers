import os
from pathlib import Path

import requests


DIRECTORY = Path('/tmp/bible')
FILENAME  = 'kjv.txt'
URL       = 'https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/eng-eng-kjv.txt'


def download_file():
    print(f'Downloading ...')
    response = requests.get(URL)
    os.makedirs(DIRECTORY, exist_ok=True)
    print(f'Pouring into {DIRECTORY / FILENAME} ...')
    with open(DIRECTORY / FILENAME, 'w') as f: f.write(response.text)
    print(f'File {FILENAME} is ready!')


if __name__ == '__main__':
    download_file()

