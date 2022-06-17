import torch
from torch import nn
import time
import torchtext
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import io
from collections import Counter
from typing import List, Tuple



class Reader:

    def __init__(self, files, lowercase=True, min_freq=0, vectorizer=None):
        self.vectorizer = vectorizer if vectorizer else self._vectorizer
        x = Counter()
        y = Counter()
        for file_name in files:
            if file_name is None:
                continue
            print(file_name)
            df=pd.read_csv(file_name,delimiter='\t',names=['Word','POS','NP','NER'],skiprows=[0])
            df=df.dropna()
            x.update(df.Word.to_list())
            y.update(df.NER.to_list())

        # build vocab
        x = dict(filter(lambda cnt: cnt[1] >= min_freq, x.items()))
        alpha = list(x.keys())
        # alpha.sort()
        self.vocab = {w: i+1 for i, w in enumerate(alpha)}
        self.vocab['[PAD]'] = 0

        self.labels = list(y.keys())
        # self.labels.sort()



        # self.lowercase = lowercase
        # self.tokenizer = tokenizer
        # build_vocab = vectorizer is None
        # self.vectorizer = vectorizer if vectorizer else self._vectorizer
        # x = Counter()
        # y = Counter()
        # for file_name in files:
        #     if file_name is None:
        #         continue
        #     with open(file_name, encoding='utf-8', mode='r') as f:
        #         for line in f:
        #             words = line.split()
        #             y.update(words[0])
        #
        #             if build_vocab:
        #                 words = self.tokenizer(' '.join(words[1:]))
        #                 words = words if not self.lowercase else [w.lower() for w in words]
        #                 x.update(words)
        # self.labels = list(y.keys())
        #
        # if build_vocab:
        #     x = dict(filter(lambda cnt: cnt[1] >= min_freq, x.items()))
        #     alpha = list(x.keys())
        #     alpha.sort()
        #     self.vocab = {w: i+1 for i, w in enumerate(alpha)}
        #     self.vocab['[PAD]'] = 0
        #
        # self.labels.sort()

    def extract_sent(self, df):
        sentences=[]
        labels=[]
        label=[]
        sentence=[]
        for word,tag in zip(df.Word,df.NER):
            label.append(tag)
            sentence.append(word)
            if word =='.':
                labels.append(label)
                sentences.append(sentence)
                sentence=[]
                label=[]
        return sentences, labels


    def _vectorizer(self, words: List[str]) -> List[int]:
        return [self.vocab.get(w, 0) for w in words]

    def load(self, filename: str) -> TensorDataset:
        label2index = {l: i+1 for i, l in enumerate(self.labels)}
        label2index['[PAD]'] = 0
        xs = []
        lengths = []
        ys = []
        df=pd.read_csv(filename,delimiter='\t',names=['Word','POS','NP','NER'],skiprows=[0])
        df=df.dropna()
        sentences, label_sets=self.extract_sent(df)

        for sentence, label_set in zip(sentences,label_sets):
            ys.append(torch.tensor(list(label2index[label] for label in label_set), dtype=torch.long))
            words=sentence # just remind that sentence are already list of words
            vec = self.vectorizer(words)
            lengths.append(len(vec))
            xs.append(torch.tensor(vec, dtype=torch.long))
        x_tensor = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        y_tensor= torch.nn.utils.rnn.pad_sequence(ys, batch_first=True)
        # y_tensor q= torch.tensor(ys, dtype=torch.long)
        return TensorDataset(x_tensor, lengths_tensor, y_tensor)




def init_embeddings(vocab_size, embed_dim, unif):
    return np.random.uniform(-unif, unif, (vocab_size, embed_dim))


class EmbeddingsReader:

    @staticmethod
    def from_text(filename, vocab, unif=0.25):

        with io.open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.rstrip("\n ")
                values = line.split(" ")

                if i == 0:
                    # fastText style
                    if len(values) == 2:
                        weight = init_embeddings(len(vocab), values[1], unif)
                        continue
                    # glove style
                    else:
                        weight = init_embeddings(len(vocab), len(values[1:]), unif)
                word = values[0]
                if word in vocab:
                    vec = np.asarray(values[1:], dtype=np.float32)
                    weight[vocab[word]] = vec
        if '[PAD]' in vocab:
            weight[vocab['[PAD]']] = 0.0

        embeddings = nn.Embedding(weight.shape[0], weight.shape[1])
        embeddings.weight = nn.Parameter(torch.from_numpy(weight).float())
        return embeddings, weight.shape[1]

    @staticmethod
    def from_binary(filename, vocab, unif=0.25):
        def read_word(f):

            s = bytearray()
            ch = f.read(1)

            while ch != b' ':
                s.extend(ch)
                ch = f.read(1)
            s = s.decode('utf-8')
            # Only strip out normal space and \n not other spaces which are words.
            return s.strip(' \n')

        vocab_size = len(vocab)
        with io.open(filename, "rb") as f:
            header = f.readline()
            file_vocab_size, embed_dim = map(int, header.split())
            weight = init_embeddings(len(vocab), embed_dim, unif)
            if '[PAD]' in vocab:
                weight[vocab['[PAD]']] = 0.0
            width = 4 * embed_dim
            for i in range(file_vocab_size):
                word = read_word(f)
                raw = f.read(width)
                if word in vocab:
                    vec = np.fromstring(raw, dtype=np.float32)
                    weight[vocab[word]] = vec
        embeddings = nn.Embedding(weight.shape[0], weight.shape[1])
        embeddings.weight = nn.Parameter(torch.from_numpy(weight).float())
        return embeddings, embed_dim
