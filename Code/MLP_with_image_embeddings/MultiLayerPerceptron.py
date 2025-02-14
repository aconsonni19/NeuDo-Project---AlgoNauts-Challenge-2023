import torch.nn as nn
# ===================================================================================
# Base MLP class
# ===================================================================================
class BrainResponseMLP(nn.Module):
    def __init__(self, name, input_dim=512, hidden_dims=[1024, 2048, 4096], output_dim=39548, dropout=0.2,
                 batch_norm=True, activation=nn.ReLU):
        super(BrainResponseMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.residuals = []  # Track which layers should have residuals

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            linear_layer = nn.Linear(prev_dim, hidden_dim)
            self.layers.append(linear_layer)

            if prev_dim == hidden_dim:  # Residual possible if dimensions match
                self.residuals.append(len(self.layers) - 1)  # Store layer index

            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(activation())
            self.layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim  # Update for next layer

        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.name = name

    def forward(self, x):
        prev_x = None  # Store for residual connections
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear) and i in self.residuals:
                x = x + layer(x)  # Apply residual
            else:
                x = layer(x)
            if isinstance(layer, nn.Linear):  # Store last linear layer output
                prev_x = x
        return self.output_layer(x)

# ===================================================================================
# Functions that return all available MLP configurations
# ===================================================================================
def get_all_available_mlp(input_dim=512, output_dim=39548):
    """
    Generator function to create and return MLP models one at a time.
    This prevents memory allocation for all models at once.
    """
    model_configs = [
        ("Base MLP", [512, 1024, 2048], 0.00, False),
        ("Base MLP Batch Normalized", [512, 1024, 2048], 0.00, True),
        ("Deep MLP", [512, 1024, 2048, 1024, 512], 0.00, False),
        ("Dropout Regularized MLP", [512, 1024, 2048], 0.5, False),
        ("Batch Normalized Shallow MLP", [512, 1024], 0.01, True),
        ("Batch and Dropout Normalized MLP", [512, 1024, 2048], 0.5, True),
        ("Leaky ReLU MLP", [512, 1024, 2048], 0.4, False, nn.LeakyReLU),
        ("GELU MLP", [512, 1024, 2048], 0.4, False, nn.GELU)
    ]

    for config in model_configs:
        # Ensure all configs have the correct number of elements
        if len(config) == 4:  # Missing activation function
            name, hidden_dims, dropout, batch_norm = config
            activation = nn.ReLU  # Default activation
        else:
            name, hidden_dims, dropout, batch_norm, activation = config

        yield BrainResponseMLP(name, input_dim, hidden_dims, output_dim, dropout, batch_norm, activation)
