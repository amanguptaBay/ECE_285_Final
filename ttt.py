import torch.nn as nn

class TTT_System(nn.Module):
    def __init__(self, F, A, B):
        super().__init__()
        self.F = F
        self. A = A
        self.B = B
        
    def forward(self, x):        
        out = self.F(x)
        return self.A(out), self.B(out)