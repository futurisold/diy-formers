import regex as re
from ftfy import fix_text
from torch.utils.data import Dataset


class WMTDataset(Dataset):
    def __init__(self, src_path: str, tgt_path: str):
        self.src = open(src_path, 'r').readlines()
        self.tgt = open(tgt_path, 'r').readlines()
        assert len(self.src) == len(self.tgt), "Source and target files must have the same number of lines"

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

    def __len__(self): return len(self.src)

    def __getitem__(self, idx): return self._parse_pair(self.src[idx], self.tgt[idx])

    def _parse_pair(self, src_line: str, tgt_line: str):
        src_line = re.findall(self.pattern, fix_text(src_line))
        tgt_line = re.findall(self.pattern, fix_text(tgt_line))

        return src_line, tgt_line

