import torch
import torch.nn as nn


class MaxBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        xm, _ = x.max(dim=1, keepdim=True)
        x = self.proj(x - xm)
        return x


class Encoder(nn.Module):
    def __init__(self, x_dim, d_dim, z1_dim):
        super().__init__()
        self.phi = nn.Sequential(
            MaxBlock(x_dim, d_dim),
            nn.Tanh(),
            MaxBlock(d_dim, d_dim),
            nn.Tanh(),
            MaxBlock(d_dim, d_dim),
            nn.Tanh(),
        )
        self.ro = nn.Sequential(
            nn.Linear(d_dim, d_dim),
            nn.Tanh(),
            nn.Linear(d_dim, z1_dim),
        )

    def forward(self, x):
        x = self.phi(x)
        x, _ = x.max(dim=1)
        z1 = self.ro(x)
        return z1


class Decoder(nn.Module):
    def __init__(self, x_dim, z1_dim, z2_dim, h_dim=512):
        super().__init__()
        self.fc = nn.Linear(z1_dim, h_dim)
        self.fu = nn.Linear(z2_dim, h_dim, bias=False)
        self.dec = nn.Sequential(
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, x_dim),
        )

    def forward(self, z1, z2):
        x = self.fc(z1) + self.fu(z2)
        o = self.dec(x)
        return o


class Generator(nn.Module):
    def __init__(self, x_dim=3, d_dim=256, z1_dim=256, z2_dim=10):
        super().__init__()
        self.z2_dim = z2_dim
        self.enc = Encoder(x_dim, d_dim, z1_dim)
        self.dec = Decoder(x_dim, z1_dim, z2_dim)

    def encode(self, x):
        z1 = self.enc(x).unsqueeze(dim=1)
        return z1

    def decode(self, z1, B, N, device):
        z2 = torch.randn((B, N, self.z2_dim)).to(device)
        o = self.dec(z1, z2)
        return o

    def forward(self, x):
        z1 = self.encode(x)
        o = self.decode(z1, x.size(0), x.size(1), x.device)
        return o, z1


class Discriminator(nn.Module):
    def __init__(self, x_dim=3, z1_dim=256, d_dim=256, h_dim=1024, o_dim=1):
        super().__init__()
        self.fc = nn.Linear(z1_dim, h_dim)
        self.fu = nn.Linear(x_dim, h_dim, bias=False)
        self.d1 = nn.Sequential(
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim - z1_dim),
        )
        self.sc = nn.Linear(z1_dim, h_dim)
        self.su = nn.Linear(h_dim - z1_dim, h_dim, bias=False)
        self.d2 = nn.Sequential(
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim - z1_dim),
        )
        self.tc = nn.Linear(z1_dim, h_dim)
        self.tu = nn.Linear(h_dim - z1_dim, h_dim, bias=False)
        self.d3 = nn.Sequential(
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, o_dim),
        )

    def forward(self, x, z1):
        y = self.fc(z1) + self.fu(x)
        o = self.d1(y)
        y = self.sc(z1) + self.su(o)
        o = self.d2(y)
        y = self.tc(z1) + self.tu(o)
        o = self.d3(y)
        return o
