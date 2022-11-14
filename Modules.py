import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, dim_sequence: int, dim_bottleneck: int = 2, num_heads: int = 1, dim_positional_encoding: int = 0, num_layers: int = 1) -> None:
        super().__init__()

        # TODO Comment faire ça propre ?
        self.dim_sequence = dim_sequence
        self.dim_bottleneck = dim_bottleneck
        self.num_heads = num_heads
        self.dim_positional_encoding = dim_positional_encoding
        self.num_layers = num_layers

        self.positional_encoding = nn.Parameter(torch.rand((dim_sequence, dim_positional_encoding), requires_grad=True)) # TODO see if requires_grad=True is needed here
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
        # TODO si on regarde comment la mémoire se remplit, pourquoi ya 2 gros saut au premier batch à l'étape encoder et decoder, et ensuite plus du tout de sauts ?
        # Pourquoi il utilise pas la mémoire où est stocké l'input ? Pourquoi seulement 2 ? 
        # Pourquoi de la mémoire serait prise quand on fait le forward pass ? Tous les paramètres du module ont déjà été crée au moment de l'instantiation, non ?
        # donc c'est le batch en sortie ? Mais ça m'a l'air + gros que la taille du batch.
        # Peut-être parce que TransformerEncoder ça rend une copie, alors que les autres layers pas, mais genre Linear ça rend pas une copie ?
        # Si oui, pourquoi on voudrait que les transformers rendent une copie mais pas les Linear ?
        # Et pourquoi les modules ne font pas des in-place operations ?
        # Et pourquoi avec l'ancienne méthode il arrivait, alors que je mettais des full encoder-decoder-transformers, plusieurs layers, batch size 64, etc ???

    # TODO Pourquoi le fait de ne pas définir une fonction à chaque fois ça prend pas moins de mémoire ? 
    # Est-ce que si je fais b = a, ça ajoute juste un drapeau "b" à la variable pointée par "a" ? 
    # Est-ce que ça dépend si a est mutable ou immutable ? Si oui, c'est quoi le sens de faire ça, là ça n'a rien à voir avec le fait de modifier
    def forward_(self, input: torch.Tensor):
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
        x = torch.cat((input_unsqueezed, positional_encoding), dim=-1) # (N, S) -> (N, S, 1+dim_positional_encoding)
        del input_unsqueezed, positional_encoding
        # Tiling to make num_heads copies of the embeddings
        # See this if there is an answer
        # https://stackoverflow.com/questions/74355156/expanding-a-non-singleton-dimension-in-pytorch-but-without-copying-data-in-memo
        x = torch.tile(x, (self.num_heads,)) # (N, S, 1+dim_positional_encoding) -> (N, S, num_heads*(1+dim_positional_encoding))
        x = self.encoder(x) # (N, S, num_heads*(1+dim_positional_encoding)) -> (N, S, num_heads*(1+dim_positional_encoding))
        input_encoded_shape = x.shape
        x = torch.flatten(x, start_dim=-2) # (N, S, num_heads*(1+dim_positional_encoding)) -> (N, S*num_heads*(1+dim_positional_encoding))
        x = self.bottleneck(x) # (N, S*num_heads*(1+dim_positional_encoding)) -> (N, S*num_heads*(1+dim_positional_encoding))
        # TODO verify that reshape does not mix elements compared to flatten, i.e. reshape(flatten) = identity
        x = torch.reshape(x, input_encoded_shape) # (N, S*num_heads*(1+dim_positional_encoding)) -> (N, S, num_heads*(1+dim_positional_encoding))
        x = self.encoder(x) # (N, S, num_heads*(1+dim_positional_encoding)) -> (N, S, num_heads*(1+dim_positional_encoding))
        output = self.condense_heads(x).squeeze() # (N, S, num_heads*(1+dim_positional_encoding)) -> (N, S)
        return output