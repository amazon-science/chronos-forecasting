from torch import nn
from torch.nn import CrossEntropyLoss
import torch

from chronosx.injection_blocks.basic_modules import InjectionBlock, FeedForwardNN


class OutputInjectionBlock(InjectionBlock):
    name = "output_injection_block"

    def __init__(
        self,
        hidden_dim: int = 256,
        model_dim: int = 512,
        num_covariates: int = 0,
        num_layers: int = 1,
        vocab_size: int = 4096,
    ):
        super(OutputInjectionBlock, self).__init__()

        self.hidden_dim = hidden_dim
        self.model_dim = model_dim
        self.num_covariates = num_covariates
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # self.num_layers = num_layers
        self.concat_dim = 2 * self.hidden_dim
        self.cov_out = nn.Linear(self.num_covariates, self.hidden_dim)
        self.concat_layer = FeedForwardNN(
            self.concat_dim, [self.hidden_dim] * self.num_layers, self.vocab_size
        )
        self.hidden_state_out = nn.Linear(self.model_dim, self.hidden_dim)
        self.loss_fct = CrossEntropyLoss(ignore_index=-100)

    def compute_loss(self, logits, labels):
        labels = labels.to(logits.device)
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def forward(self, future_covariates, labels, logits, last_hidden_state):
        x = self.hidden_state_out(last_hidden_state)

        if self.generating:
            x_cov = future_covariates[:, self.counter].unsqueeze(1)
        else:
            x_cov = future_covariates

        if x_cov.flatten().isnan().sum() > 0:
            print(1)

        x_cov = self.cov_out(x_cov)  # -> here is the problem
        x = torch.concatenate([x, x_cov], axis=-1)

        x = nn.ReLU()(x)
        logits = self.concat_layer(x) + logits

        if self.training:
            loss = self.compute_loss(logits, labels)
        else:
            loss = None
            self.counter += 1

        return logits, loss
