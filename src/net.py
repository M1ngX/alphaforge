import torch
import torch.nn as nn


class NetP(nn.Module):
    def __init__(self, action_size, hidden_size, dropout=0.1):
        super(NetP, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(action_size, 96, kernel_size=(1, 3), padding=(0, 1)),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 8)),
            nn.Conv2d(96, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 4))
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 4, hidden_size),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, return_latent=False):
        x = x.permute(0, 2, 1).unsqueeze(2)
        x = self.convs(x)

        x = x.reshape(x.size(0), -1)
        latent = self.fc1(x)
        x = self.fc2(latent)
        out = x.reshape(-1)
        return (out, latent) if return_latent else out
    
    def reset_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) < 2:
                    nn.init.xavier_normal_(param.unsqueeze(0))
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)


# class NetP(nn.Module):
#     def __init__(self, action_size, hidden_size, num_layers, dropout):
#         super(NetP, self).__init__()
#         self.lstm = nn.LSTM(
#             input_size=action_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_size, 1)
#         )

#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#     def forward(self, x, return_latent=False):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         latent, _ = self.lstm(x, (h0, c0))
#         out = self.fc(latent[:, -1, :])
#         out = out.reshape(-1)
#         return (out, latent) if return_latent else out

#     def reset_params(self):
#         for name, param in self.named_parameters():
#             if 'weight' in name:
#                 if len(param.shape) < 2:
#                     nn.init.xavier_normal_(param.unsqueeze(0))
#                 else:
#                     nn.init.xavier_normal_(param)
#             elif 'bias' in name:
#                 nn.init.constant_(param, 0.0)


class NetG(nn.Module):
    def __init__(self, action_size, hidden_size, seq_len):
        super(NetG, self).__init__()
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        
        self.linear = nn.Linear(action_size, hidden_size * 8)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, hidden_size // 2, (4, 1), (2, 1), (1, 0), bias=True),
            nn.BatchNorm2d(hidden_size // 2),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(hidden_size // 2, hidden_size // 4, (4, 1), (2, 1), (1, 0), bias=True),
            nn.BatchNorm2d(hidden_size // 4),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(hidden_size // 4, hidden_size // 8, (4, 1), (2, 1), (1, 0), bias=True),
            nn.BatchNorm2d(hidden_size // 8),
            nn.LeakyReLU(),
        )
        
        self.conv = nn.Sequential(
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Conv2d(hidden_size // 8, hidden_size // 16, (3, 1), (1, 1), 0, bias=True),
            nn.BatchNorm2d(hidden_size // 16),
            nn.LeakyReLU(),
            
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Conv2d(hidden_size // 16, hidden_size // 32, (3, 1), (1, 1), 0, bias=True),
            nn.BatchNorm2d(hidden_size // 32),
            nn.LeakyReLU(),

            nn.Conv2d(hidden_size // 32, action_size, (1, 1), (1, 1), 0, bias=True),
        )
        
        self.seq_adjust = nn.Linear(64, seq_len)
                
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.linear(x)
        x = x.view(batch_size, self.hidden_size, 8, 1)

        x = self.deconv(x)
        x = self.conv(x)
        
        x = x.squeeze(-1)
        x = self.seq_adjust(x).transpose(1, 2)
        return x
    
    def reset_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) < 2:
                    nn.init.xavier_normal_(param.unsqueeze(0))
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)


# class NetG(nn.Module):
#     def __init__(self, action_size, hidden_size, seq_len, num_layers, dropout):
#         super(NetG, self).__init__()
#         self.embedding = nn.Linear(action_size, hidden_size)
#         self.lstm = nn.LSTM(
#             input_size=hidden_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_size, action_size),
#         )

#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.seq_len = seq_len

#     def forward(self, z):
#         z = self.embedding(z)
#         h = torch.zeros(self.num_layers, z.size(0), self.hidden_size).to(z.device)
#         c = torch.zeros(self.num_layers, z.size(0), self.hidden_size).to(z.device)
        
#         outputs = []
#         input = z.unsqueeze(1)  # (N, 1, hidden_size)

#         for t in range(self.seq_len):
#             output, (h, c) = self.lstm(input, (h, c))
#             outputs.append(output.squeeze(1))
#             input = output

#         out = torch.stack(outputs, dim=1)
#         out = self.fc(out)
#         return out

#     def reset_params(self):
#         for name, param in self.named_parameters():
#             if 'weight' in name:
#                 if len(param.shape) < 2:
#                     nn.init.xavier_normal_(param.unsqueeze(0))
#                 else:
#                     nn.init.xavier_normal_(param)
#             elif 'bias' in name:
#                 nn.init.constant_(param, 0.0)