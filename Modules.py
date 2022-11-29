import torch
from torch import nn
import math

class TransformerAutoencoder(nn.Module):
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


class RandomConnexionsTransformerAutoencoder(nn.Module):
    """TODO expliquer l'idée des random connexions (et changer le nom maybe), expliquer le fait que le random
    est choisi seulement à la construction du model, et ensuite il est gardé tout le long."""
    def __init__(self, dim_input: int, dim_embedding: int, dim_bottleneck: int = 2, num_heads: int = 1, dim_positional_encoding: int = 0, num_layers: int = 1) -> None:
        # TODO mettre valeur par defaut à dim_embedding quand plus besoin que ce soit un diviseur de dim_input
        super().__init__()

        self.dim_input = dim_input
        self.dim_embedding = dim_embedding
        self.dim_bottleneck = dim_bottleneck
        self.num_heads = num_heads
        self.dim_positional_encoding = dim_positional_encoding
        self.num_layers = num_layers

        # The input of the transformer is of size (N, S, E), where 
        # N is the batch size,
        # S = ceil(dim_input / dim_embedding)
        # E = num_heads * (dim_embedding + dim_positional_encoding)
        # We multiply d_model by nhead because we want multiple attention heads without dividing the embedding into smaller parts
        # We concatenate positional encoding istead of summing to embedding because embedding is of low dimension 
        # TODO try with addition instead of concatenation

        # Right now, dim_embedding must be a divisor of dim_input #TODO implement general case
        assert dim_input % dim_embedding == 0, \
            f"dim_embedding must be a divisor of dim_input, got dim_embedding={dim_embedding} and dim_input={dim_input}."

        self.dim_sequence = math.ceil(dim_input / dim_embedding)
        self.dim_positionally_encoded_attention_headed_embedding = num_heads * (dim_embedding + dim_positional_encoding)

        # The random connexions are doing as follow:
        # In the forward function, the input is shuffled and then we reshape the input with shape (dim_sequence, dim_embedding)
        # so that we have dim_sequence chunks of lenght dim_embeddings
        # Just before the output of the autoencoder, the shuffle is inverted and the matrix is reshaped (S, E) -> (S*E,) = (I,)
        # To perform the shuffeling and inverse shuffeling, we compute a permutation and inverse permuation of the index from
        # 0 to dim_input (excluded)
        self.permutation = nn.Parameter(torch.randperm(dim_input), requires_grad=False)
        self.permutation = nn.Parameter(torch.arange(dim_input), requires_grad=False)
        # https://stackoverflow.com/questions/66832716/how-to-quickly-inverse-a-permutation-by-using-pytorch
        self.inverse_permutation = nn.Parameter(torch.empty_like(self.permutation), requires_grad=False)
        self.inverse_permutation[self.permutation] = torch.arange(self.permutation.size(0))
        # TODO pas ouf les noms, trouver mieux
        # TODO en fait la permutation inverse elle sert à rien, le decoder transformer il fait les bails tout seul maybe, à tester
        
        self.positional_encoding = nn.Parameter(torch.rand((self.dim_sequence, dim_positional_encoding), requires_grad=True))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim_positionally_encoded_attention_headed_embedding, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerEncoderLayer(d_model=self.dim_positionally_encoded_attention_headed_embedding, nhead=num_heads, batch_first=True) # TODO besoin d'un 2ème ?
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=num_layers)
        self.bottleneck = nn.Sequential(
            nn.Linear(self.dim_sequence * self.dim_positionally_encoded_attention_headed_embedding, dim_bottleneck), 
            nn.Linear(dim_bottleneck, self.dim_sequence * self.dim_positionally_encoded_attention_headed_embedding))
        self.condense_heads = nn.Linear(self.dim_positionally_encoded_attention_headed_embedding, dim_embedding)

    def forward(self, input: torch.Tensor):
        """Input is of shape (N, I) or (I,), where N is the batch dimension and I is the input dimension.
        
        The input corresponds for example to a tensor with all the pixels of an image, or all the neurons of a layer.
        """

        assert input.shape[-1] == self.dim_input, \
            f"Expected dimension of input {self.dim_input}, got {input.shape[-1]}."
        # TODO verifier que on a que (N, I) ou (I,) comme shape et pas un truc à + que 2 dimensions

        N = 1 if len(input.shape) == 1 else input.size(0)
        # TODO peut-etre une bonne idee de unsqueeze une dimension au début si N=1 dans tous les cas, faut voir, c'est peut-etre plus clair
        # parce que comme ça ya pas besoin de se poser la question genre est-ce que c'est (I,) ou (N, I), c'est (N, I) pour sûr


        # We create the sequence/embedding matrix with the method described above
        # To each data of the batch, we apply the random permutation
        input_permuted = torch.index_select(input, -1, self.permutation) # (N, I) -> (N, I) or (I,) -> (I,)
        # then we construct the token matrix
        # (N, I) -> (N, S, dim_embedding) or (I,) -> (N=1, S, dim_embedding)
        input_permuted_matrix = input_permuted.reshape(-1, self.dim_sequence, self.dim_embedding)

        # (S, dim_positional_encoding) -> (N, S, dim_positional_encoding)
        positional_encoding = self.positional_encoding.expand((N, -1, -1))
        # We concatenate each positional encoding with each token
        # (N, S, dim_embedding) + (N, S, dim_positional_encoding) -> (N, S, dim_embedding + dim_positional_encoding)
        input_positioned = torch.cat((input_permuted_matrix, positional_encoding), dim=-1)
        # Tiling to make num_heads copies of the embeddings
        # (N, S, dim_embedding + dim_positional_encoding) -> (N, S, E = num_heads * (dim_embedding + dim_positional_encoding))
        input_tiled = torch.tile(input_positioned, (self.num_heads,))
        input_encoded = self.encoder(input_tiled) # (N, S, E) -> (N, S, E)
        input_encoded_flattened = torch.flatten(input_encoded, start_dim=-2) # (N, S, E) -> (N, S*E)
        output_encoded_flattened = self.bottleneck(input_encoded_flattened) # (N, S*E) -> (N, S*E)
        output_encoded = torch.reshape(output_encoded_flattened, input_encoded.shape) # (N, S*E) -> (N, S, E)
        output_decoded = self.decoder(output_encoded) # (N, S, E) -> (N, S, E)
        output_permuted_matrix = self.condense_heads(output_decoded) # (N, S, E) -> (N, S, dim_embedding)
        output_permuted = output_permuted_matrix.reshape(input_permuted.shape) # (N, S, dim_embedding) -> (N, I)
        output = torch.index_select(output_permuted, -1, self.inverse_permutation).squeeze() # (N, I) -> (N, I) or (N, I) -> (I,) if N=1
        return output


class LinearAutoencoder(nn.Module):
    """Autoencoder made of only linear layers.
    
    At each deeper layer of the encoder, the dimension is reduced of `step`, until it is within `step` from `dim_bottleneck`.
    Then there is a bottleneck made of two linear layers with middle dimension `dim_bottleneck`
    The decoder is the mirrored version of the encoder.
    All the Linear layers of the encoder and decoder have activation function specified by `activation_function`."""

    def __init__(self, dim_input: int, dim_bottleneck: int, step: int = 3, activation_function: str = 'ReLU') -> None:
        """activation function: ReLU or Tanh"""
         
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
        self.bottleneck = nn.Sequential(nn.Linear(self.dimensions[-1], dim_bottleneck), 
                                        nn.Linear(dim_bottleneck, self.dimensions[-1]))
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


class TransformerIdentity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # https://stackoverflow.com/questions/60908827/why-pytorch-nn-module-cuda-not-moving-module-tensor-but-only-parameters-and-bu
        self.positional_encoding = nn.Parameter(torch.rand((28), requires_grad=True)) # TODO see if requires_grad=True is needed here
        self.flatten = nn.Flatten()
        self.transformer = nn.Transformer(
            d_model=28, 
            nhead=1, 
            num_encoder_layers=3, 
            num_decoder_layers=3, 
            batch_first=True) # https://discuss.pytorch.org/t/why-is-sequence-batch-features-the-default-instead-of-bxsxf/8244

    def forward(self, input):
        """Takes a batch of 28*28 pictures (N,28,28) # TODO verifier que c'est pas (N,1,28,28) que donne le dataloader
        We want to have (N,S,E)=(N,28,28), i.e. each row is a 'token' and the whole picture is a sentence."""
        input = torch.add(input, self.positional_encoding) # picture += positional encoding
        # input = self.flatten(input) # (N,28,28) -> (N,784)
        # input = torch.unsqueeze(input, 2) # (N,784) -> (N,784,1)
        input = torch.squeeze(input) # TODO lié au TODO d'en haut (N,1,28,28)->(N,28,28)
        output = self.transformer(input, input)
        # output = torch.reshape(output, input.shape) # (N,1,784) -> (N,28,28)
        return output