'''
Concrete IO class for Stage 4 text classification dataset + GloVe embeddings
'''

from local_code.base_class.dataset import dataset
import os
import re
import glob
import numpy as np
from collections import Counter


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None   # 'text_classification'
    glove_folder_name = 'glove.2024.wikigiga.50d'
    embedding_dim = 50
    max_len = 200
    max_vocab_size = 20000

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def _tokenize(self, text):
        text = text.replace('<br />', ' ').lower()
        return re.findall(r"[a-z0-9']+", text)

    def _read_split(self, split_dir):
        texts, labels = [], []
        label_map = {'neg': 0, 'pos': 1}
        if not os.path.isdir(split_dir):
            print(f'[WARNING] フォルダが見つかりません: {split_dir}')
            return texts, np.array(labels, dtype=np.int64)

        # 二重フォルダ(train/train/posなど)でも自動で pos / neg を探す
        for root, dirs, files in os.walk(split_dir):
            folder_name = os.path.basename(root)
            if folder_name in label_map:
                label = label_map[folder_name]
                for fname in sorted(files):
                    if fname.startswith('.') or fname == 'Thumbs.db':
                        continue
                    fpath = os.path.join(root, fname)
                    if os.path.isfile(fpath):
                        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                            texts.append(self._tokenize(f.read()))
                            labels.append(label)

        if len(texts) == 0:
            print(f'[WARNING] {split_dir} の中に pos/neg データが見つかりませんでした。中身: {os.listdir(split_dir)}')
        return texts, np.array(labels, dtype=np.int64)

    def _build_vocab(self, train_texts):
        counter = Counter()
        for tokens in train_texts:
            counter.update(tokens)
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in counter.most_common(self.max_vocab_size - 2):
            vocab[word] = len(vocab)
        return vocab

    def _texts_to_sequences(self, texts, vocab):
        unk_id = vocab['<UNK>']
        X = np.zeros((len(texts), self.max_len), dtype=np.int64)
        for i, tokens in enumerate(texts):
            ids = [vocab.get(tok, unk_id) for tok in tokens[:self.max_len]]
            if ids:
                X[i, :len(ids)] = ids
        return X

    def _load_glove(self, vocab):
        glove_dir = os.path.join(self.dataset_source_folder_path, self.glove_folder_name)
        txt_files = glob.glob(os.path.join(glove_dir, '*.txt'))
        if not txt_files:
            raise FileNotFoundError(f'No GloVe .txt file found in {glove_dir}')
        glove_path = txt_files[0]
        print(f'loading GloVe embeddings from {os.path.basename(glove_path)}...')

        rng = np.random.default_rng(2)
        emb_matrix = rng.normal(0.0, 0.1, size=(len(vocab), self.embedding_dim)).astype(np.float32)
        emb_matrix[0] = 0.0  # <PAD> is all zeros

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
        print('loading text classification data...')
        base_dir = os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name)
        train_texts, y_train = self._read_split(os.path.join(base_dir, 'train'))
        test_texts, y_test = self._read_split(os.path.join(base_dir, 'test'))

        vocab = self._build_vocab(train_texts)
        X_train = self._texts_to_sequences(train_texts, vocab)
        X_test = self._texts_to_sequences(test_texts, vocab)
        emb_matrix = self._load_glove(vocab)

        print(f'loaded train: {X_train.shape}, test: {X_test.shape}')
        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test},
            'embedding_matrix': emb_matrix
        }