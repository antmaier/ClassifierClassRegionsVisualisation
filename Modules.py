import torch
from torch import nn
import math

class Autoencoder(nn.Module):
    def __init__(self, dim_sequence: int, dim_bottleneck: int = 2, num_heads: int = 1, dim_positional_encoding: int = 0, num_layers: int = 1) -> None:
        super().__init__()

        # TODO Comment faire ça propre ? Dataclass or TypedDict?
        self.dim_sequence = dim_sequence
        self.dim_bottleneck = dim_bottleneck
        self.num_heads = num_heads
        self.dim_positional_encoding = dim_positional_encoding
        self.num_layers = num_layers

        self.positional_encoding = nn.Parameter(torch.rand((dim_sequence, dim_positional_encoding), requires_grad=True))
        # We multiply d_model by nhead because we want multiple attention heads without dividing the embedding into smaller parts
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=num_heads * (1 + dim_positional_encoding), nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerEncoderLayer(d_model=num_heads * (1 + dim_positional_encoding), nhead=num_heads, batch_first=True) # TODO besoin d'un 2ème ?
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=num_layers)
        self.bottleneck = nn.Sequential(
            nn.Linear(dim_sequence * num_heads * (1 + dim_positional_encoding), dim_bottleneck), 
            nn.Linear(dim_bottleneck, dim_sequence * num_heads * (1 + dim_positional_encoding)))
        self.condense_heads = nn.Linear(num_heads*(1+dim_positional_encoding), 1)
        
    
    def forward(self, input: torch.Tensor):
        """Input is of shape (N,S) or (S,), where S = dim_sequence""" # TODO nomenclature S, dim_trucs, N, etc
        assert input.shape[-1] == self.dim_sequence, \
            f"Expected dimension of sequence {self.dim_sequence}, got {input.shape[-1]}."

        # TODO say that (...) -> (...) c'est la shape avant/après, et dire que ça s'applique avec le N devant et sans
        
        input_unsqueezed = input.unsqueeze(-1) # (N, S) -> (N, S, 1)
        # TODO Not sure if optimizer applied to this new view will work correctly
        # https://medium.com/dive-into-ml-ai/caution-while-using-nn-parameter-in-pytorch-3ef3de5a6557
        positional_encoding = self.positional_encoding.expand(tuple(size for size in input.size()) + (-1,)) # (S, dim_positional_encoding) -> (N, S, dim_positional_encoding)
        # We concatenate each positional encoding with each "token". 
        # We concatenate instead of adding because here the token is of dim 1, which is very small
        input_position = torch.cat((input_unsqueezed, positional_encoding), dim=-1) # (N, S) -> (N, S, 1+dim_positional_encoding)
        # Tiling to make num_heads copies of the embeddings
        # See this if there is an answer
        # https://stackoverflow.com/questions/74355156/expanding-a-non-singleton-dimension-in-pytorch-but-without-copying-data-in-memo
        input_tiled = torch.tile(input_position, (self.num_heads,)) # (N, S, 1+dim_positional_encoding) -> (N, S, num_heads*(1+dim_positional_encoding))
        input_encoded = self.encoder(input_tiled) # (N, S, num_heads*(1+dim_positional_encoding)) -> (N, S, num_heads*(1+dim_positional_encoding))
        input_encoded_flattened = torch.flatten(input_encoded, start_dim=-2) # (N, S, num_heads*(1+dim_positional_encoding)) -> (N, S*num_heads*(1+dim_positional_encoding))
        output_encoded_flattened = self.bottleneck(input_encoded_flattened) # (N, S*num_heads*(1+dim_positional_encoding)) -> (N, S*num_heads*(1+dim_positional_encoding))
        # TODO verify that reshape does not mix elements compared to flatten, i.e. reshape(flatten) = identity
        output_encoded = torch.reshape(output_encoded_flattened, input_encoded.shape) # (N, S*num_heads*(1+dim_positional_encoding)) -> (N, S, num_heads*(1+dim_positional_encoding))
        output_decoded = self.decoder(output_encoded) # (N, S, num_heads*(1+dim_positional_encoding)) -> (N, S, num_heads*(1+dim_positional_encoding))
        output = self.condense_heads(output_decoded).squeeze() # (N, S, num_heads*(1+dim_positional_encoding)) -> (N, S)
        return output

class RandomConnexionsAutoencoder(nn.Module):
    def __init__(self, dim_input: int, dim_embedding: int = 10, dim_bottleneck: int = 2, num_heads: int = 1, dim_positional_encoding: int = 0, num_layers: int = 1) -> None:
        super().__init__()

        self.dim_input = dim_input
        self.dim_embedding = dim_embedding
        self.dim_bottleneck = dim_bottleneck
        self.num_heads = num_heads
        self.dim_positional_encoding = dim_positional_encoding
        self.num_layers = num_layers

        # The input of the transformer is of size (N, S, E), where 
        # N is the batch size,
        # S = ceil(dim_sequence / dim_embedding)
        # E = n_heads * (dim_embedding + dim_positional_encoding)
        # We multiply d_model by nhead because we want multiple attention heads without dividing the embedding into smaller parts

        dim_sequence = math.ceil(dim_sequence / dim_embedding)
        
        self.positional_encoding = nn.Parameter(torch.rand((dim_sequence, dim_positional_encoding), requires_grad=True))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_sequence, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerEncoderLayer(d_model=dim_sequence, nhead=num_heads, batch_first=True) # TODO besoin d'un 2ème ?
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=num_layers)
        self.bottleneck = nn.Sequential(
            nn.Linear(dim_sequence, dim_bottleneck), 
            nn.Linear(dim_bottleneck, dim_sequence))
        self.condense_heads = nn.Linear(num_heads*(1+dim_positional_encoding), 1)

class LinearAutoencoder(nn.Module):
    """Autoencoder made of only linear layers.
    
    At each deeper layer of the encoder, the dimension is reduced of `step`, until it is within `step` from `dim_bottleneck`.
    If it does not exactly attains `dim_bottleneck`, one Linear layer with output `dim_bottleneck` is added.
    The decoder is the mirrored version of the encoder.
    All the Linear layers have activation function specified by `activation_function`."""

    def __init__(self, dim_input: int, dim_bottleneck: int, step: int = 3, activation_function: str = 'ReLU') -> None:
        super().__init__()

        assert dim_input > 0, \
            f"dim_input must be strictly greater than 0, got {dim_input}."
        assert dim_bottleneck > 0, \
            f"dim_bottleneck must be strictly greater than 0, got {dim_bottleneck}."
        assert step > 0, \
            f"step must be striclty greater than 0, got {step}."
        assert activation_function in ['ReLU', 'Tanh'], \
            f"Non-supported activation function, got {activation_function}."

        # List of dimensions, from dim_input to dim_bottleneck TODO mettre à jour
        # self.dimensions = list(range(dim_input, dim_bottleneck - 1, -step))
        self.dimensions = list(range(dim_input, dim_bottleneck, -step))
        # if self.dimensions[-1] > dim_bottleneck: # True if (dim_input - dim_bottleneck) is not a multiple of step
        #     self.dimensions.append(dim_bottleneck)
        # self.encoder = nn.ModuleList([nn.Linear(self.dimensions[i], self.dimensions[i+1]) for i in range(len(self.dimensions[:-1]))])
        self.encoder = nn.ModuleList([nn.Linear(dim, dim - step) for dim in self.dimensions[:-1]])
        self.bottleneck = nn.Sequential(nn.Linear(self.dimensions[-1], dim_bottleneck), nn.Linear(dim_bottleneck, self.dimensions[-1]))
        # self.decoder = nn.ModuleList([nn.Linear(self.dimensions[i], self.dimensions[i+1]) for i in reversed(range(len(self.dimensions[:-1])))])
        self.decoder = nn.ModuleList([nn.Linear(dim, dim + step) for dim in reversed(self.dimensions[1:])])
        if activation_function == 'ReLU':
            self.activation_function = nn.ReLU()
        elif activation_function == 'Tanh':
            self.activation_function = nn.Tanh()

    def forward(self, x):
        for module in self.encoder:
            x = self.activation_function(module(x))
        x = self.bottleneck(x)
        for module in self.decoder:
            x = self.activation_function(module(x))
        return x



