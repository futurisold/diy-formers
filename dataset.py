from collections import Counter

import regex as re
from ftfy import fix_text
from torch.utils.data import Dataset


class KJBible(Dataset):
    def __init__(self, src: str, vocab_size: int) -> None:
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

        self.vocab_size = vocab_size
        self.corpus = self.__load_data(src) # corpus as text
        self.byte_encoder = self.__bpe_encoding()

    def __load_data(self, src: str) -> str: return fix_text(' '.join(open(src, 'r', encoding='utf-8', errors='replace').readlines()))

    def __tokenize(self, corpus: list) -> list: return re.findall(self.pattern, corpus, re.IGNORECASE)

    def __bpe_encoding(self) -> dict:
        vocab = self.__init_vocab()
        bigrams = Counter()
        while (tokens := sum(map(len, vocab))) > self.vocab_size:
            for token in vocab:
                for bigram in zip(token[:-1], token[1:]):
                    if bigram in bigrams: bigrams[bigram] += 1
                    else: bigrams[bigram] = 1

            # most common bigram
            most_common = bigrams.most_common(1)[0][0]
            vocab = self.__merge(most_common, vocab)

            print(f'compression ratio: {tokens / self.vocab_size:.4f} -> 1', end='\r')

        return bigrams

    def __merge(self, most_common: tuple, vocab: dict) -> dict:
        new_vocab = {}

        for token in vocab:
            if len(token) == 1: new_vocab[token] = vocab[token]
            else:
                new_token = self.__new_token(token, most_common)
                if new_token in new_vocab: new_vocab[new_token] += vocab[token]
                else: new_vocab[new_token] = vocab[token]

        return new_vocab

    def __new_token(self, token: tuple, bigram: tuple) -> tuple:
        new_token = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and token[i:i+2] == bigram:
                new_token.append(token[i] + token[i+1])
                i += 2
            else:
                new_token.append(token[i])
                i += 1

        return tuple(new_token)


    def __init_vocab(self):
        tokens = self.__tokenize(self.corpus)[:-1] # last is garbage
        vocab = []
        for token in tokens:
            token = list(token) + ['</w>']
            vocab.append(tuple(token))

        return Counter(vocab)

