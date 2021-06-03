from torch import nn
import torch
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, feature_size, num_classes, sent_len, conv_sizes, dropout = 0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.sent_len = sent_len
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, feature_size, kernel_size) 
             for kernel_size in conv_sizes]
        )
        self.pools = nn.ModuleList(
            [nn.MaxPool1d(sent_len - kernel_size + 1) 
             for kernel_size in conv_sizes]
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(len(conv_sizes) * feature_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, text, text_lengths):
        # text = text.permute([0,2,1])
        x = self.embedding(text)
        # print(text)
        # print(x)
        # print(x.shape)
        x = F.pad(x, (0,0,0, self.sent_len - x.shape[1]))
        x = x.permute([0,2,1])
        # print(x.shape)
        x = [conv(x) for conv in self.convs]
        # print(x[0].shape, x[1].shape, x[2].shape)
        x = [nn.ReLU()(y) for y in x]
        # print(x[0].shape, x[1].shape, x[2].shape)
        x = [self.pools[i](x[i]) for i in range(len(x))]
        x = torch.cat(x, dim=1)
        x = torch.squeeze(x, dim=2)
        x = self.fc(x)
        return x