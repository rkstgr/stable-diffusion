import torch
import torch.nn as nn


class MdctLoss(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=1.0):
        super().__init__()
        self.kl_weight = kl_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def forward(self, inputs, reconstructions, posteriors, **kwargs):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss = nll_loss + kl_loss * self.kl_weight
        split = kwargs["split"]
        log = {
            f"{split}/total_loss": loss.clone().detach().mean(),
            f"{split}/logvar": self.logvar.detach(),
            f"{split}/kl_loss": kl_loss.detach().mean(),
            f"{split}/nll_loss": nll_loss.detach().mean(),
        }
        return loss, log