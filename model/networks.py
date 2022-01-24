import torch
import torch.nn as nn

from .layers import AttentiveBlock, InitialSet, ISAB
from .ops import get_module, masked_fill


class ElementwiseMLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
    ):
        super().__init__()
        self.out = nn.Linear(dim_in, dim_out)

    def forward(self, x, x_mask=None):  # [B, N, C]
        x = masked_fill(x, x_mask)
        x = self.out(x)
        return x


class InducedNetwork(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
    ):
        super().__init__()
        self.net = ElementwiseMLP(dim_in, dim_out)

    def forward(self, x):
        return self.net(x)


class EncoderBlock(AttentiveBlock):
    """ISAB in Set Transformer"""

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln, slot_att):
        super().__init__(dim_in, dim_out, num_heads, num_inds, ln, slot_att)


class DecoderBlock(AttentiveBlock):
    """ABL (Attentive Bottleneck Layer)"""

    def __init__(
        self,
        dim_in,
        dim_out,
        dim_z,
        num_heads,
        num_inds,
        ln,
        slot_att,
        cond_prior=True,
    ):
        super().__init__(dim_in, dim_out, num_heads, num_inds, ln, slot_att)
        self.cond_prior = cond_prior
        if cond_prior:
            self.prior = InducedNetwork(
                dim_out,
                2 * dim_z,
            )
        else:
            self.register_parameter(
                name="prior", param=nn.Parameter(torch.randn(1, num_inds, 2 * dim_z))
            )  # [1, M, 2Dz]
            nn.init.xavier_uniform_(self.prior)
        self.posterior = InducedNetwork(
            dim_out,
            2 * dim_z,
        )
        self.fc = nn.Linear(dim_z, dim_out)

    def compute_prior(self, h):
        """
        Sample from prior
        :param h: Tensor([B, M, D])
        :return: Tensor([B, M, Dz])
        """
        bs, num_inds, dim_in = h.shape
        if self.cond_prior:  # [B, M, 2Dz]
            prior = self.prior(h)
        else:
            prior = self.prior.repeat(bs, 1, 1)
        mu = prior[..., : prior.shape[-1] // 2]  # [B, M, Dz]
        logvar = prior[..., prior.shape[-1] // 2 :].clamp(-4.0, 3.0)
        eps = torch.randn(mu.shape).to(h)
        z = mu + torch.exp(logvar / 2.0) * eps  # [B, M, Dz]
        return z, mu, logvar

    def compute_posterior(self, mu, logvar, bottom_up_h, h=None):
        """
        Estimate residual posterior parameters from prior parameters and top-down features
        :param mu: Tensor([B, M, D])
        :param logvar: Tensor([B, M, D])
        :param bottom_up_h: Tensor([B, M, D])
        :param h: Tensor([B, M, D])
        :return: Tensor([B, M, Dz]), Tensor([B, M, Dz])
        """
        bs, num_inds, dim_in = bottom_up_h.shape
        assert self.num_inds == num_inds
        bottom_up_h = bottom_up_h + h if h is not None else bottom_up_h
        posterior = self.posterior(bottom_up_h)
        mu2 = posterior[..., : posterior.shape[-1] // 2]  # [B, M, Dz]
        logvar2 = posterior[..., posterior.shape[-1] // 2 :].clamp(-4.0, 3.0)
        sigma = torch.exp(logvar / 2.0)
        sigma2 = torch.exp(logvar2 / 2.0)
        eps = torch.randn(mu.shape).to(mu)
        z = (mu + mu2) + (sigma * sigma2) * eps
        kl = -0.5 * (logvar2 + 1.0 - mu2.pow(2) / sigma.pow(2) - sigma2.pow(2)).view(
            mu.shape[0], -1
        ).sum(
            dim=-1
        )  # [B,]
        return z, kl, mu2, logvar2

    def broadcast_latent(self, z, h, x, x_mask=None):
        return self.broadcast(self.fc(z), x, x_mask)  # No residual


class SetVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        max_outputs,
        init_dim,
        n_mixtures,
        n_layers,
        z_dim,
        z_scales,
        hidden_dim,
        num_heads,
        slot_att,
        isab_inds,
        ln,
    ):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.init_set = InitialSet(
            init_dim,
            n_mixtures,
            hidden_dim,
            max_outputs,
        )
        self.encoder = nn.ModuleList(
            [
                EncoderBlock(
                    hidden_dim,
                    hidden_dim,
                    num_heads,
                    z_scales[::-1][i],
                    ln,
                    slot_att,
                )
                for i in range(n_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(
                    hidden_dim,
                    hidden_dim,
                    z_dim,
                    num_heads,
                    z_scales[i],
                    ln,
                    slot_att,
                    cond_prior=i > 0,
                )
                for i in range(n_layers)
            ]
        )
        self.output = nn.Linear(hidden_dim, input_dim)

    def bottom_up(self, x, x_mask):
        """Deterministic bottom-up encoding
        :param x: Tensor([B, N, Di])
        :param x_mask: BoolTensor([B, N])
        :return: List([Tensor([B, M, D])]), List([Tensor([H, B, N, M]), Tensor([H, B, N, M])])
        """
        x = self.input(x)  # [B, N, D]
        features, alphas = [], []
        for layer in get_module(self.encoder):
            x, h, alpha1, alpha2 = layer(
                x, x_mask
            )  # [B, N, D], [B, M, D], [H, B, N, M], [H, B, N, M]
            features.append(h)
            alphas.append((alpha1, alpha2))
        return features, alphas

    def top_down(self, cardinality, bottom_up_h):
        """Stochastic top-down decoding
        :param cardinality: Tensor([B,])
        :param bottom_up_h: List([Tensor([B, M, D])]) in top-down order
        :return:
        """
        o, o_mask = self.init_set(cardinality)
        alphas, posteriors, kls = [], [(o, None, None)], []
        for idx, layer in enumerate(get_module(self.decoder)):
            h, alpha1 = layer.project(o, o_mask)
            _, mu, logvar = layer.compute_prior(h)
            z, kl, mu2, logvar2 = layer.compute_posterior(
                mu, logvar, bottom_up_h[idx], None if idx == 0 else h
            )
            o, alpha2 = layer.broadcast_latent(z, h, o, o_mask)
            alphas.append((alpha1, alpha2))
            posteriors.append((z, mu2, logvar2))
            kls.append(kl)
        o = self.output(o)  # [B, N, Do]
        return o, o_mask, posteriors, kls, alphas

    def forward(self, x, x_mask):
        """Bidirectional inference
        :param x: Tensor([B, N, Di])
        :param x_mask: BoolTensor([B, N])
        :return: Tensor([B, N, Do]), Tensor([B, N]), List([Tensor([H, B, N, M]), Tensor([H, B, N, M])]) * 2
        """
        features, _ = self.bottom_up(x, x_mask)
        o, o_mask, _, kls, _ = self.top_down(
            (~x_mask).sum(-1), list(reversed(features))
        )
        return o, o_mask, kls

    def sample(self, output_sizes):
        """Top-down generation
        :param output_sizes: Tensor([B,])
        :return: Tensor([B, N, Do]), Tensor([B, N]), List([Tensor([B, M, D])]),
                 List([Tensor([H, B, N, M]), Tensor([H, B, N, M])])
        """
        o, o_mask = self.init_set(output_sizes)
        priors = [(o, None, None)]
        alphas = list()
        for idx, layer in enumerate(get_module(self.decoder)):
            h, alpha1 = layer.project(o, o_mask)
            if idx == 0:
                z, mu, logvar = layer.compute_prior(h)
            z, mu, logvar = layer.compute_prior(h)
            o, alpha2 = layer.broadcast_latent(z, h, o, o_mask)
            priors.append((z, mu, logvar))
            alphas.append((alpha1, alpha2))
        o = self.output(o)  # [B, N, Do]
        return o, o_mask, priors, alphas
