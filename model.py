from torch import nn
import torch.nn.functional as F
from spikingjelly.clock_driven import rnn

class SpikingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.slstm = rnn.SpikingLSTM(input_size, hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.slstm(x)
        x = self.fc(x[-1])
        x = F.softmax(x, dim=1)
        
        return x