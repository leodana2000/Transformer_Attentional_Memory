import torch as t
from typing import List

class Transformer(t.nn.Module):
    """Transformer architecture with parallel attention heads and MLPs, additive positional embeddings, and layer-norm."""
   
    def __init__(
            self, d: int, N: int, nb_layers: int, width: int,
            parallel_heads: int, d_head: int, nb_head: int, context_window: int, 
            pi: List[t.Tensor], device: str = 'cpu',
            ) -> None:
        """
        Parameters.
        d: embedding dimension,
        N: size of the vocabulary,
        width: width of the MLP,
        depth: number of consecutive layers,
        nb_head: number of sub-heads in an attention module, it should divide d,
        nb_layers: number of layers in the Transformer,
        context_window: maximum length of a sequence of tokens, including the token to be predicted,
        pi: list of the conditional distribution for each tokens,
        device: the device on which to place the model,
        """

        assert d_head%nb_head == 0
        self.use_mlp: bool = nb_layers*width > 0 
        self.d: int = d
        self.N: int = N
        self.nb_layers: int = nb_layers
        self.width: int = width
        self.para: int = parallel_heads
        self.d_head: int = d_head
        self.nb_head: int = nb_head
        self.context_window: int = context_window
        self.n_gram: int = len(pi)
        self.pi: List[t.Tensor] = pi
        self.device: str = device

        super().__init__()

        self.word_emb = t.nn.Linear(d, N, bias=False)
        self.pos_emb = t.nn.Linear(d_head, context_window, bias=False) # Additive positional embedding

        # Implements para attention module in parallel at head layers, each containing nb_head heads.
        self.attn_seq = t.nn.Sequential(
            *[
                t.nn.Sequential(
                    *[t.nn.MultiheadAttention(d_head, nb_head, batch_first=True, bias=False) for _ in range(parallel_heads)]
                )
            for _ in range(nb_layers)]
        )

        self.linear_compress = t.nn.Linear(d, d_head, bias=False)
        self.linear_decompress = t.nn.Linear(d_head, d, bias=False)

        # Implements MLPs with fixed width and depth at each layer.
        # Doesn't implements if 0 layers of 0 hidden layers or 0 width.
        if self.use_mlp:
            self.mlp_seq = t.nn.Sequential(
                *[t.nn.Sequential(
                    t.nn.Linear(d, width, bias=True), 
                    t.nn.GELU(), 
                    t.nn.Linear(width, d, bias=False),
                ) for _ in range(nb_layers)]
            )
        else:
            self.mlp_seq = t.nn.Sequential(*[t.nn.Sequential()]*len(self.attn_seq))

        self.unemb = t.nn.Linear(d, N, bias=False)

        attn_mask = t.tril(t.ones((self.context_window-1, self.context_window-1), device=self.device)) == 0
        self.register_buffer('attn_mask', attn_mask)


    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Computes the forward pass of the Transformer.
        """
        res = self.word_emb.weight[x]
        pos = self.pos_emb.weight[:x.shape[1]].unsqueeze(0)

        d=self.d
        d_head=self.d_head

        for para_attn, mlp in zip(self.attn_seq, self.mlp_seq):

            compressed_res = self.linear_compress(res) + pos if d!=d_head else res+pos
            para_res = sum(attn(compressed_res, compressed_res, compressed_res, attn_mask=self.attn_mask)[0] for attn in para_attn)
            decompressed_res = self.linear_decompress(para_res) if d!=d_head else para_res
            res = decompressed_res.add_(res) 
                
            if self.use_mlp:
                mlp_out = mlp(res)
                res = mlp_out.add_(res)
            
        logits = self.unemb(res)
        return logits
    

class AoT(Transformer):
    def __init__(self, d: int, N: int, nb_layers: int, parallel_heads: int, d_head: int, nb_head: int, context_window: int, pi: List[t.Tensor], device: str='cpu') -> None:
        super().__init__(d, N, nb_layers, 0, parallel_heads, d_head, nb_head, context_window, pi, device=device)


class Low_rank(t.nn.Module):
    def __init__(self, d: int, N: int, context_window: int, pi: List[t.Tensor], device: str = 'cpu') -> None:
        self.use_mlp: bool = False 
        self.d: int = d
        self.N: int = N
        self.nb_layers: int = 0
        self.width: int = 0
        self.para: int = 0
        self.d_head: int = 0
        self.nb_head: int = 0
        self.context_window: int = context_window
        self.n_gram: int = len(pi)
        self.pi: List[t.Tensor] = pi
        self.device: str = device

        super().__init__()
        self.word_emb = t.nn.Linear(d, N**(self.n_gram-1), bias=False)
        self.unemb = t.nn.Linear(d, N, bias=False)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = x[:, :-1] + x[:, 1:]*self.N

        #concatenates anything in first position since we don't care about i-th prediction for i < n-gram - 1
        x = t.cat([t.zeros(x.shape[0], 1).to(t.int).to(self.device), x], dim=1) 

        logits = self.unemb(self.word_emb.weight[x])
        return logits