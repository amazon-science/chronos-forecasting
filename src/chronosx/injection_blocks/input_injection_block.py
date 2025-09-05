from torch import nn
import torch

from chronosx.injection_blocks.basic_modules import InjectionBlock, FeedForwardNN


class InputInjectionBlock(InjectionBlock):
    name = "input_injection_block"

    def __init__(
        self,
        hidden_dim: int = 256,
        model_dim: int = 512,
        num_covariates: int = 0,
        num_layers: int = 1,
    ):
        super(InputInjectionBlock, self).__init__()

        self.hidden_dim = hidden_dim
        self.model_dim = model_dim
        self.num_covariates = num_covariates
        self.num_layers = num_layers

        self.cov_in = nn.Linear(self.num_covariates, self.hidden_dim)
        self.concat_dim = self.hidden_dim * 2

        self.concat_layer = FeedForwardNN(
            self.concat_dim, [self.hidden_dim] * self.num_layers, self.model_dim
        )
        self.emb_in = nn.Linear(self.model_dim, self.hidden_dim)

    def forward(self, input_embeds, past_covariates, is_decoder=False):

        x = self.emb_in(input_embeds)
        if self.generating and is_decoder:
            past_covariates = past_covariates[:, self.counter, :].unsqueeze(1)
            self.counter += 1
        x_cov = self.cov_in(past_covariates)
        x = torch.cat([x, x_cov], axis=-1)
        x = nn.ReLU()(x)

        return input_embeds + self.concat_layer(x)
