from torch import nn
import math


class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FeedForwardNN, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class InjectionBlock(nn.Module):
    name = "generic"

    def __init__(
        self,
    ):
        super(InjectionBlock, self).__init__()
        self.counter = 0
        self.generating = False

    def restart_generator_counter(self):
        self.counter = 0
        self.generating = True

    def initialize_modules(self, modules=None):
        if modules is None:
            modules = self.modules()
        for module in modules:
            if isinstance(module, nn.Linear):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                    elif "bias" in name:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        nn.init.uniform_(param, -bound, bound)
