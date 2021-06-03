from torch import nn
import torch
import torch.nn.functional as F

class TextMLP(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes, sent_len, dropout):
        super(TextMLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.sent_len = sent_len
        self.emb_dim = emb_dim
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(sent_len * emb_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    def forward(self, text, text_lengths):
        x = self.embedding(text)
        x = F.pad(x, (0,0,0, self.sent_len - x.shape[1]))
        x = torch.reshape(x, [-1, self.sent_len * self.emb_dim])
        x = self.fc(x)
        return x