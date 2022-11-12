import torch
from torch import nn

class GeneralTransformer(nn.Module):
    def __init__(self, num_heads: int = 8, dim_positional_encoding: int = 0, num_layers: int = 6) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=num_heads * (1 + dim_positional_encoding), nhead=num_heads)] * num_layers)

    def forward(self, input: torch.Tensor):
        """Data has to be of shape (N,S,1+dim_positional_encoding), where N=batch size, S=sequence size."""
        # Expand last dim to make num_heads copies of the embeddings
        # See this if there is an answer
        # https://stackoverflow.com/questions/74355156/expanding-a-non-singleton-dimension-in-pytorch-but-without-copying-data-in-memo
        x = torch.tile(input, (self.num_heads,))
        # TODO en fait ça marche pas de faire ça, le transformer en sortie va me rendre un objet où la dimension
        # des embbedings est trop grande. Pas clair quoi faire avec ça.
        # DONC go refaire le transformer à la main
        for module in self.layers:
            x = module(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, dim_sequence: int, dim_bottleneck: int = 2, num_heads: int = 8, dim_positional_encoding: int = 0, num_layers: int = 6) -> None:
        super().__init__()

        # TODO Comment faire ça propre ?
        self.dim_sequence = dim_sequence
        self.dim_bottleneck = dim_bottleneck
        self.num_heads = num_heads
        self.dim_positional_encoding = dim_positional_encoding
        self.num_layers = num_layers

        # We multiply d_model by nhead because we want multiple attention heads without dividing the embedding into smaller parts
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=num_heads * (1 + dim_positional_encoding), nhead=num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.bottleneck = nn.Sequential(nn.Linear(dim_sequence, dim_bottleneck), nn.Linear(dim_bottleneck, dim_sequence))
        
    def forward(self, input):
        """Input is of shape (N,S,1+dim_positional_encoding) or (S,1+dim_positional_encoding)"""
        assert input.shape[-2] == self.sequence_dim, \
            f"Expected sequence dimension {self.sequence_dim}, got {input.shape[-2]}."
        assert input.shape[-1] == 1 + self.dim_positional_encoding, \
            f"Expected embedding dimension {1 + self.dim_positional_encoding}, got {input.shape[-1]}."

        # Broadcasting to make num_heads copies of the embeddings
        # See this if there is an answer
        # https://stackoverflow.com/questions/74355156/expanding-a-non-singleton-dimension-in-pytorch-but-without-copying-data-in-memo
        x = torch.tile(input, (self.num_heads,))
        x = self.encoder(x)
        x = self.bottleneck(x)


        x = torch.reshape(self.bottleneck(self.flatten(x)), x.shape)
        output = self.decoder(x)
        return output
