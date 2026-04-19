import torch
import torch.nn as nn

class ChartNet(nn.Module):
    def __init__(self, fc_feature=600, audio_feature=500, hidden_dim=512, num_layers=2, output_dim=5):
        """
        Modified ChartNet for Clone Hero Expert Drums.
        Output dimension bumped down from 16 to 5 representing independent labels for:
        (0:Kick, 1:Red, 2:Yellow, 3:Blue, 4:Green).
        We use AdaptiveAvgPool to guarantee matrix dimension safety across resolutions.
        """
        super(ChartNet, self).__init__()

        self.audio_feature = audio_feature
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input: [batch, seq_len, 1, 128, 87]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((30, 20))
        
        self.fc1 = nn.Linear(in_features=30*20, out_features=fc_feature)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=fc_feature, out_features=audio_feature)
        self.dropout1 = nn.Dropout(p=0.1)

        self.bilstm = nn.LSTM(audio_feature, hidden_dim, num_layers, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        # Note: Do not add Sigmoid here, we use BCEWithLogitsLoss during training.

    def forward(self, x):
        # x is expected to be [batch, seq_len, 1, 128, max_frames]
        batch_size, seq_len, c, h, w = x.shape
        
        x = x.view(batch_size * seq_len, c, h, w)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.dropout1(x)
        
        x = x.view(batch_size, seq_len, self.audio_feature)
        
        output, _ = self.bilstm(x)
        
        return self.fc(output)
