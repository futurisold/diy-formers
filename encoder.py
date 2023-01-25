import pickle
from collections import Counter
from pathlib import Path

import regex as re
from ftfy import fix_text
from tqdm import trange


class Encoder:
    '''
    @arg src:             path to the directory containing the corpus
    @arg from_pretrained: whether to load the byte encoder and vocab from the src directory
    '''
    def __init__(self, src: str, from_pretrained: bool) -> None:
        # demistifying the regex behind GPT-2 encoder: https://github.com/openai/gpt-2/blob/master/src/encoder.py#L53
        self.__pattern =  ''.join([
            "'s|",                 # matches contractions such as "he's" -> [he, 's]
            "'t|",                 # matches contractions such as "can't" -> [can, 't]
            "'re|",                # matches contractions such as "they're" -> [they, 're]
            "'ve|",                # matches contractions such as "we've" -> [we, 've]
            "'ll|",                # matches contractions such as "we'll" -> [we, 'll]
            "'d|",                 # matches contractions such as "she'd" -> [she, 'd]
            "'m|"                  # matches contractions such as "i'm" -> [i, 'm]
            " ?\p{L}+|",           # matches one or more letters, with an optional preceding space
            " ?\p{N}+|",           # matches one or more digits, with an optional preceding space
            " ?[^\s\p{L}\p{N}]+|", # matches one or more non-letter, non-digit characters, with an optional preceding space
            "\s+(?!\S)|",          # matches one or more spaces that are not followed by a non-space
            "\s+"                  # matches one or more spaces
        ])
        self.__merges = 40_000 # aiming for what GPT-1 had
        # corpus as text
        self.corpus = self.__load_data(Path(src) / 'corpus.txt')
        # corpus as tokens
        if not from_pretrained:
            self.byte_encoder, self.vocab = self.__bpe_encoding()
            # dump the vocab and byte_encoder if encoder was used for the first time
            self.__dump(self.byte_encoder, Path(src) / 'byte_encoder.pkl')
            self.__dump(self.vocab, Path(src) / 'vocab.pkl')
        else:
            self.byte_encoder = pickle.load(open(Path(src) / 'byte_encoder.pkl', 'rb'))
            self.vocab = pickle.load(open(Path(src) / 'vocab.pkl', 'rb'))
        # mappings
        self.t2i = {token: i for i, token in enumerate(set(self.vocab))}
        self.i2t = {i: token for i, token in enumerate(set(self.vocab))}

    def encode(self, text: str) -> list:
        tokens = re.findall(self.__pattern, text, re.IGNORECASE)
        encoded = []
        for token in tokens:
            if len(token) == 1:
                encoded.append(token)
                continue

            token = list(token) + ['</w>']

            while True:
                bigrams = list(zip(token[:-1], token[1:]))
                hits = [(bigram, self.byte_encoder[bigram]) for bigram in bigrams if bigram in self.byte_encoder]
                if not hits: break

                # least common bigram
                least_common = min(hits, key=lambda x: x[1])[0]
                token = self.__new_token(token, least_common)

            encoded.extend(token)

        return encoded

    def decode(self, tokens: list) -> str: return ''.join(tokens).replace('</w>', '').strip()

    def __load_data(self, src: str) -> str: return fix_text(' '.join(open(src, 'r', encoding='utf-8', errors='replace').readlines()))

    def __bpe_encoding(self) -> dict:
        vocab = self.__init_vocab()
        bigrams = Counter()
        byte_encoder = {}
        for i in trange(self.__merges):
            for token in vocab:
                for bigram in zip(token[:-1], token[1:]):
                    if bigram in bigrams: bigrams[bigram] += 1
                    else: bigrams[bigram] = 1

            # most common bigram
            most_common = bigrams.most_common(1)[0][0]
            # merge most common bigram
            vocab = self.__merge(most_common, vocab)
            # update byte_encoder
            byte_encoder[most_common] = i

        # flatten the vocab
        vocab = [subtoken for token in vocab for subtoken in token]

        return byte_encoder, vocab

    def __merge(self, most_common: tuple, vocab: dict) -> dict:
        new_vocab = {}

        for token in vocab:
            new_token = self.__new_token(token, most_common)
            new_vocab[new_token] = vocab[token]

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
        tokens = re.findall(self.__pattern, self.corpus, re.IGNORECASE)
        vocab = []
        for token in tokens:
            if ''.join(token).count('\n') > 1: continue # remove paragraphs
            token = list(token) + ['</w>']
            vocab.append(tuple(token))

        return Counter(vocab)

    def __dump(self, obj: object, name: str) -> None: pickle.dump(obj, open(name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

