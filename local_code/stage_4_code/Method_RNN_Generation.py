'''
Stage 4 RNN/LSTM/GRU model (PyTorch) for text generation
'''

from local_code.base_class.method import method
import re
import numpy as np
import torch
from torch import nn


class Method_RNN_Generation(method, nn.Module):
    data = None

    # ---- hyper-parameters ----
    rnn_type = 'rnn'                # 'rnn', 'lstm', or 'gru'
    hidden_dim = 128
    num_layers = 1
    dropout = 0.2
    freeze_embedding = False
    loss_name = 'cross_entropy'
    optimizer_name = 'adam'
    learning_rate = 3e-3
    weight_decay = 0.0
    batch_size = 32
    max_epoch = 25
    device_name = 'auto'
    # ----------------------------

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
        vocab_size, emb_dim = emb_matrix.shape
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
            dropout=rnn_dropout
        )
        self.drop_layer = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.hidden_dim, vocab_size)

    def get_device(self):
        if self.device_name == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            if torch.backends.mps.is_available():
                return torch.device('mps')
            return torch.device('cpu')
        return torch.device(self.device_name)

    def forward(self, x, hidden=None):
        emb = self.drop_layer(self.embedding(x))
        out, hidden = self.rnn(emb, hidden)
        logits = self.fc(self.drop_layer(out))
        return logits, hidden

    def compute_loss(self, logits, y):
        if self.loss_name == 'cross_entropy':
            return nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=0
            )
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
                logits, _ = self.forward(xb)
                loss = self.compute_loss(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                mask = (yb != 0)
                valid_tokens = mask.sum().item()
                total_loss += loss.item() * valid_tokens
                correct += ((logits.argmax(dim=-1) == yb) & mask).sum().item()
                count += valid_tokens

            epoch_loss = total_loss / max(count, 1)
            epoch_acc = correct / max(count, 1)
            self.history['train_loss'].append(epoch_loss)
            self.history['train_acc'].append(epoch_acc)
            print('Epoch: %3d  Loss: %.4f  Train accuracy: %.4f' % (epoch + 1, epoch_loss, epoch_acc))

    def test(self, X):
        device = self.get_device()
        self.eval()
        X_t = torch.tensor(np.asarray(X), dtype=torch.long)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_t), 128):
                logits, _ = self.forward(X_t[i:i + 128].to(device))
                preds.append(logits.argmax(dim=-1).cpu())
        return torch.cat(preds, dim=0).numpy()

    def generate_text(self, prompt, max_new_tokens=20, temperature=0.8):
        device = self.get_device()
        self.eval()
        vocab = self.data['vocab']
        idx2word = self.data['idx2word']
        unk_id = vocab['<UNK>']
        eos_id = vocab['<EOS>']

        prompt_tokens = ['<BOS>'] + re.findall(r"[a-z0-9']+|[.,!?;]", prompt.lower().strip())
        ids = [vocab.get(tok, unk_id) for tok in prompt_tokens]
        generated = list(prompt_tokens[1:])

        with torch.no_grad():
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            logits, hidden = self.forward(inp)
            next_logits = logits[0, -1] / max(temperature, 1e-5)
            next_logits[0] = -1e9  # disallow <PAD>
            next_logits[unk_id] = -1e9  # disallow <UNK>

            for _ in range(max_new_tokens):
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
                if next_id == eos_id:
                    break
                word = idx2word.get(next_id, '')
                if word not in ('<BOS>', '<PAD>', '<UNK>'):
                    generated.append(word)
                inp = torch.tensor([[next_id]], dtype=torch.long, device=device)
                logits, hidden = self.forward(inp, hidden)
                next_logits = logits[0, -1] / max(temperature, 1e-5)
                next_logits[0] = -1e9
                next_logits[unk_id] = -1e9

        text = ' '.join(generated)
        text = re.sub(r'\s+([.,!?;])', r'\1', text)
        return text

    def run(self):
        print('method running...')
        self.build_model()
        print(self)
        print('device:', self.get_device())
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_all = self.test(self.data['test']['X'])
        true_all = np.asarray(self.data['test']['y'])
        mask = (true_all != 0)
        return {'pred_y': pred_all[mask], 'true_y': true_all[mask]}