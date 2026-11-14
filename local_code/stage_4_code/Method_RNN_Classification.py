'''
Stage 4 RNN/LSTM/GRU model (PyTorch) for text classification
'''

from local_code.base_class.method import method
import numpy as np
import torch
from torch import nn


class Method_RNN_Classification(method, nn.Module):
    data = None

    # ---- hyper-parameters (override these from the script) ----
    rnn_type = 'rnn'                # 'rnn', 'lstm', or 'gru'
    hidden_dim = 128
    num_layers = 1
    bidirectional = False
    output_dim = 2                  # 0: neg, 1: pos
    dropout = 0.3
    freeze_embedding = False
    loss_name = 'cross_entropy'
    optimizer_name = 'adam'
    learning_rate = 1e-3
    weight_decay = 0.0
    batch_size = 64
    max_epoch = 10
    device_name = 'auto'            # 'cpu', 'cuda', 'mps', or 'auto'
    # ------------------------------------------------------------

    def __init__(self, mName=None, mDescription=None):
        nn.Module.__init__(self)
        method.__init__(self, mName, mDescription)
        self.history = {'train_loss': [], 'train_acc': []}
        self.embedding = None
        self.rnn = None
        self.drop_layer = None
        self.fc = None

    def build_model(self):
        emb_matrix = torch.tensor(self.data['embedding_matrix'], dtype=torch.float32)
        emb_dim = emb_matrix.shape[1]
        self.embedding = nn.Embedding.from_pretrained(
            emb_matrix, freeze=self.freeze_embedding, padding_idx=0
        )

        rnn_dropout = self.dropout if self.num_layers > 1 else 0.0
        rnn_cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[self.rnn_type.lower()]
        self.rnn = rnn_cls(
            input_size=emb_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=rnn_dropout
        )

        self.drop_layer = nn.Dropout(self.dropout)
        fc_in = self.hidden_dim * (2 if self.bidirectional else 1)
        self.fc = nn.Linear(fc_in, self.output_dim)

    def get_device(self):
        if self.device_name == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            if torch.backends.mps.is_available():
                return torch.device('mps')
            return torch.device('cpu')
        return torch.device(self.device_name)

    def forward(self, x):
        lengths = (x != 0).sum(dim=1).clamp(min=1).cpu()
        emb = self.drop_layer(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )

        if self.rnn_type.lower() == 'lstm':
            _, (h_n, _) = self.rnn(packed)
        else:
            _, h_n = self.rnn(packed)

        if self.bidirectional:
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            last_hidden = h_n[-1]

        logits = self.fc(self.drop_layer(last_hidden))
        return logits

    def compute_loss(self, logits, y):
        if self.loss_name == 'cross_entropy':
            return nn.functional.cross_entropy(logits, y)
        raise ValueError('unknown loss: ' + self.loss_name)

    def make_optimizer(self):
        params = self.parameters()
        lr, wd = self.learning_rate, self.weight_decay
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        if self.optimizer_name == 'adamw':
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        raise ValueError('unknown optimizer: ' + self.optimizer_name)

    def train_model(self, X, y):
        device = self.get_device()
        self.to(device)
        X_t = torch.tensor(np.asarray(X), dtype=torch.long)
        y_t = torch.tensor(np.asarray(y), dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t),
            batch_size=self.batch_size, shuffle=True
        )
        optimizer = self.make_optimizer()
        self.history = {'train_loss': [], 'train_acc': []}

        for epoch in range(self.max_epoch):
            self.train()
            total_loss, correct, count = 0.0, 0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = self.forward(xb)
                loss = self.compute_loss(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * len(yb)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                count += len(yb)

            epoch_loss = total_loss / count
            epoch_acc = correct / count
            self.history['train_loss'].append(epoch_loss)
            self.history['train_acc'].append(epoch_acc)
            print('Epoch: %3d  Loss: %.4f  Train accuracy: %.4f' % (epoch + 1, epoch_loss, epoch_acc))

    def test(self, X):
        device = self.get_device()
        self.eval()
        X_t = torch.tensor(np.asarray(X), dtype=torch.long)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_t), 256):
                logits = self.forward(X_t[i:i + 256].to(device))
                preds.append(logits.argmax(dim=1).cpu())
        return torch.cat(preds).numpy()

    def run(self):
        print('method running...')
        self.build_model()
        print(self)
        print('device:', self.get_device())
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}