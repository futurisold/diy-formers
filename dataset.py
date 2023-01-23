import time
import regex as re
from ftfy import fix_text

s = time.time()
from torch.utils.data import Dataset
e = time.time()
print(e-s)

class KJBible(Dataset):
    def __init__(self, src: str) -> None:
        self.src = open(src, 'r').readlines()

        # demistifying the regex behind GPT-2 encoder: https://github.com/openai/gpt-2/blob/master/src/encoder.py#L53
        self.pattern =  ''.join([
            "'s|",                 # matches contractions such as "he's" -> [he, 's]
            "'t|",                 # matches contractions such as "can't" -> [can, 't]
            "'re|",                # matches contractions such as "they're" -> [they, 're]
            "'ve|",                # matches contractions such as "we've" -> [we, 've]
            "'ll|",                # matches contractions such as "we'll" -> [we, 'll]
            "'d|",                 # matches contractions such as "she'd" -> [she, 'd]
            "'m|"                  # matches contractions such as "i'm" -> [i, 'm]
            " ?\p{L}+|",           # matches only words
            " ?\p{N}+|",           # matches only numbers
            " ?[^\s\p{L}\p{N}]+|", # matches everything that is not a word or a number
            "\s+(?!\S)",           # matches trailing and leading spaces
            "\s+"                  # matches multiple spaces
        ])

    def __len__(self) -> int: return len(self.src)

    def __getitem__(self, idx: int) -> list[str]: return self._parse_line(self.src[idx])

    def _parse_line(self, line: str) -> str: return re.findall(self.pattern, fix_text(line))

