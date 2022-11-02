# Pay Attention with Intention
# Transformer from Scratch, referencing- https://www.youtube.com/watch?v=U0s0f995w14
# Repo- https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py
# Also Referencing- http://peterbloem.nl/blog/transformers


import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads) -> None:
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dimension = embed_size // heads #number of elements per head vector (portion of the input to each self-attention)

        assert (self.head_dimension * heads == embed_size), "Embed Size needs to be divisible by number of Heads!"

        # Self-Attention Mappings (Queries, Keys, Values)
        self.values = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.keys = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.queries = nn.Linear(self.head_dimension, self.head_dimension, bias=False)

        self.fc_out = nn.Linear(heads*self.head_dimension, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0] #how many examples we send in at once
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1] #correspond to dims of input/output sequence?

        # Split Embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dimension)
        keys = keys.reshape(N, key_len, self.heads, self.head_dimension)
        queries = query.reshape(N, query_len, self.heads, self.head_dimension)

        # Wizardry for Matmul (with additional dimensions)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) #alt version for matrix multiplications, super cool way of handling dims
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len) 
        #can think of "query_len" as length of target sentence
        #can think of "key_len" as length of input sentence

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20")) #"small" value to represent -infinity

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3) #normalizing across "key_len" dimension (pay attention to source sentence, normalize attention to 1)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dimension)
        # attention shape: (N, heads, query_len, key_len)
        # value shape: (N, value_len, heads, heads_dim)
        # output shape: (N, query_len, heads, heads_dim) -- after einsum/matmul (then we flatten last two dims)
        #value_len and key_len always the same length/value, do multiplication across this dim (Linear Algebra) -- referred to as 'l' in einsum

        # Linear Layer
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion) -> None:
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads) #use above attention implementation
        self.norm1 = nn.LayerNorm(embed_size) #normalize across a single Layer (per example averaging, more compute than batch_norm)
        self.norm2 = nn.LayerNorm(embed_size) #

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size), #in Attention is All you need, forward_expansion is set to 4
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size), #does a transformation and then maps back to the original dimensions
        )

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x)) #skip connection -- see diagram in paper for a single transformer block (literally same thing)


class Encoder(nn.Module):
    # Init w Hyperparameters
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length          #related to the positional embedding (how long is the max sentence length)
    ) -> None:
        super(Encoder, self).__init__() 

        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions)) #position embedding is the thing that does the magic behind figuring out how words "interact"
        for layer in self.layers:
            out = layer(out, out, out, mask) #in Encoder: value, key and query are all the same (out, out, out)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device) -> None:
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        #target mask is essential! src mask is somewhat optional (prevents compute for padded sequences)
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)

        return out


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length) -> None:
        super(Decoder, self).__init__()

        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask, trg_mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, src_mask, trg_mask)

        out = self.fc_out(x) #prediction of which word is next, dependent on `trg_mask`
        return out


# Bring it Home
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256, #512 in paper?
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        # device="cuda", #originally made on non-cuda compiled version of torch
        device="cpu",
        max_length=100
    ) -> None:
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len) #triangular lower matrix (see paper and docs)
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out




# Does it work? Toy Example
if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #classic
    device = "cpu"
    x = torch.tensor([[1, 2, 4, 1, 5, 2, 8, 9, 0, 7], [8, 1, 6, 3, 8, 9, 6, 7, 3, 4]]).to(device)
    trg = torch.tensor([[4, 2, 3, 2, 9, 8, 7, 9, 1, 7], [2, 1, 6, 2, 6, 1, 6, 3, 3, 4]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10

    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
    out = model(x, trg[:, :-1])
    print(out.shape)
