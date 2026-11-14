'''
Concrete IO class for Stage 4 text generation dataset (jokes) + GloVe embeddings
'''

from local_code.base_class.dataset import dataset
import os
import re
import csv
import glob
import numpy as np
from collections import Counter


class Dataset_Loader_Generation(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None   # 'text_generation'
    glove_folder_name = 'glove.2024.wikigiga.50d'
    embedding_dim = 50
    max_len = 35
    max_vocab_size = 5000

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def _tokenize(self, text):
        text = text.lower().strip()
        tokens = re.findall(r"[a-z0-9']+|[.,!?;]", text)
        return ['<BOS>'] + tokens + ['<EOS>']

    def _read_jokes(self, data_path):
        jokes = []
        with open(data_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[1].strip():
                    tokens = self._tokenize(row[1])
                    if len(tokens) >= 4:
                        jokes.append(tokens)
        return jokes

    def _build_vocab(self, train_jokes):
        counter = Counter()
        for tokens in train_jokes:
            counter.update(tokens)
        vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        for word, _ in counter.most_common(self.max_vocab_size - 4):
            if word not in vocab:
                vocab[word] = len(vocab)
        idx2word = {idx: word for word, idx in vocab.items()}
        return vocab, idx2word

    def _jokes_to_xy(self, jokes, vocab):
        unk_id = vocab['<UNK>']
        X = np.zeros((len(jokes), self.max_len), dtype=np.int64)
        y = np.zeros((len(jokes), self.max_len), dtype=np.int64)
        for i, tokens in enumerate(jokes):
            ids = [vocab.get(tok, unk_id) for tok in tokens[:self.max_len + 1]]
            if len(ids) > 1:
                inp = ids[:-1]
                tgt = ids[1:]
                X[i, :len(inp)] = inp
                y[i, :len(tgt)] = tgt
        return X, y

    def _load_glove(self, vocab):
        glove_dir = os.path.join(self.dataset_source_folder_path, self.glove_folder_name)
        txt_files = glob.glob(os.path.join(glove_dir, '*.txt'))
        if not txt_files:
            raise FileNotFoundError(f'No GloVe .txt file found in {glove_dir}')
        glove_path = txt_files[0]
        print(f'loading GloVe embeddings from {os.path.basename(glove_path)}...')

        rng = np.random.default_rng(2)
        emb_matrix = rng.normal(0.0, 0.1, size=(len(vocab), self.embedding_dim)).astype(np.float32)
        emb_matrix[0] = 0.0  # <PAD>

        matched = 0
        with open(glove_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.rstrip().split(' ')
                if len(parts) != self.embedding_dim + 1:
                    continue
                word = parts[0]
                if word in vocab:
                    emb_matrix[vocab[word]] = np.asarray(parts[1:], dtype=np.float32)
                    matched += 1
        print(f'matched {matched}/{len(vocab)} words with GloVe.')
        return emb_matrix

    def load(self):
        print('loading text generation data...')
        data_file = os.path.join(
            self.dataset_source_folder_path, self.dataset_source_file_name, 'data'
        )
        jokes = self._read_jokes(data_file)

        rng = np.random.default_rng(2)
        indices = rng.permutation(len(jokes))
        split = int(len(jokes) * 0.8)
        train_jokes = [jokes[i] for i in indices[:split]]
        test_jokes = [jokes[i] for i in indices[split:]]

        vocab, idx2word = self._build_vocab(train_jokes)
        X_train, y_train = self._jokes_to_xy(train_jokes, vocab)
        X_test, y_test = self._jokes_to_xy(test_jokes, vocab)
        emb_matrix = self._load_glove(vocab)

        print(f'loaded train: {X_train.shape}, test: {X_test.shape}, vocab: {len(vocab)}')
        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test},
            'embedding_matrix': emb_matrix,
            'vocab': vocab,
            'idx2word': idx2word
        }