import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN_NLP(nn.Module):
    def __init__(self,
                 embed_dim=400,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes = None,
                 dropout=0.5):
        super(CNN_NLP, self).__init__()
        self.embed_dim = embed_dim
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes

        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, batch):
  
        x = batch[0]
        # x shape: (b, max_len, embed_dim)
        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # x reshaped: (b, embed_dim, max_len)
        x_reshaped = x.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))

        return logits