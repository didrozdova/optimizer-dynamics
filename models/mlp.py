import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 128, out_dim: int = 1, act: str = 'relu', use_he: bool = True):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        self.act_name = act.lower()

        if self.act_name == "relu":
            self.act = nn.ReLU()
        elif self.act_name == "leakyrelu":
            self.act = nn.LeakyReLU(negative_slope=0.01)
        elif self.act_name == "gelu":
            self.act = nn.GELU()
        elif self.act_name == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {act}")
        
        if use_he:
            self._init_weights

    def _init_weights(self):
        if self.act_name == "relu" or self.act_name == "gelu":
            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")

        elif self.act_name == "leakyrelu":
            nn.init.kaiming_normal_(
                self.fc1.weight,
                nonlinearity="leaky_relu",
                a=0.01,
            )

        elif self.act_name == "tanh":
            nn.init.xavier_normal_(self.fc1.weight)

        nn.init.zeros_(self.fc1.bias)

        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="linear")
        nn.init.zeros_(self.fc2.bias)


    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.act(z1)
        logits = self.fc2(a1)

        return logits.squeeze(-1), z1, a1