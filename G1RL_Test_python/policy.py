import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, obs_dim: int = 255, action_dim: int = 14, hidden_dims: list = [256, 256, 128]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ELU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PolicyExporter(nn.Module):
    def __init__(self, obs_dim: int = 255, action_dim: int = 14, hidden_dims: list = [256, 256, 128]):
        super().__init__()
        self.actor = SimpleMLP(obs_dim, action_dim, hidden_dims)
        self.normalizer = nn.Identity()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(obs))

    @staticmethod
    def from_jit(path: str, device: str = "cuda") -> "PolicyExporter":
        """Load weights from a TorchScript checkpoint into this nn.Module."""
        jit_model = torch.jit.load(path, map_location=device)

        # Infer dimensions from the first and last linear layers
        first_weight = dict(jit_model.named_parameters())["actor.layers.0.weight"]
        last_weight = dict(jit_model.named_parameters())["actor.layers.6.weight"]
        obs_dim = first_weight.shape[1]
        action_dim = last_weight.shape[0]

        # Infer hidden dims from intermediate linear layers
        hidden_dims = []
        for name, param in jit_model.named_parameters():
            if name.startswith("actor.layers.") and name.endswith(".weight"):
                layer_idx = int(name.split(".")[2])
                if layer_idx < 6:  # exclude last linear
                    hidden_dims.append(param.shape[0])

        model = PolicyExporter(obs_dim, action_dim, hidden_dims)
        model.load_state_dict(jit_model.state_dict())
        model.to(device)
        model.eval()
        return model
