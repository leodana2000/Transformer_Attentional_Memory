import torch as t
from typing import Dict, List
from utils import layer_norm

class Transformer(t.nn.Module):
    """Transformer architecture with parallel attention heads and MLPs, additive positional embeddings, and layer-norm."""
   
    def __init__(
            self, d: int, N: int, nb_layers: int, width: int, depth: int, 
            parallel_heads: int, d_head: int, nb_head: int, context_window: int, 
            pi: List[t.Tensor], device: str = 'cpu',
            ) -> None:
        """
        Parameters.
        d: embedding dimension,
        N: size of the vocabulary,
        width: width of the MLP,
        depth: depth of the MLP,
        nb_head: number of sub-heads in an attention module, it should divide d,
        nb_layers: number of layers in the Transformer,
        context_window: maximum length of a sequence of tokens, including the token to be predicted,
        pi: list of the conditional distribution for each tokens,
        skips: dictionary of operations to skip in the forward pass,
        """

        assert d%nb_head == 0
        self.use_mlp = nb_layers*depth*width > 0 
        self.meta_params: Dict[str, int] = {
            'd': d,
            'N': N,
            'nb_layers': nb_layers,
            'width': width,
            'depth': depth,
            'para': parallel_heads,
            'd_head': d_head,
            'nb_head': nb_head,
            'context_window': context_window,
            'n_gram': len(pi),
        }
        
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

        if d_head != d:
            self.linear_compress = t.nn.Linear(d, d_head, bias=False)
            self.linear_decompress = t.nn.Linear(d_head, d, bias=False)

        # Implements MLPs with fixed width and depth at each layer.
        # Doesn't implements if 0 layers of 0 hidden layers or 0 width.
        if self.use_mlp:
            self.mlp_seq = t.nn.Sequential(
                *[t.nn.Sequential(
                    *([t.nn.Linear(d, width, bias=True)] + [t.nn.GELU() if i%2 == 0 else t.nn.Linear(width, width, bias=True) for i in range(2*(depth-1)+1)] + [t.nn.Linear(width, d, bias=False)])
                ) for _ in range(nb_layers)]
            )
        else:
            self.mlp_seq = t.nn.Sequential(*[t.nn.Sequential()]*len(self.attn_seq))

        self.unemb = t.nn.Linear(d, N, bias=False)


    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Computes the forward pass of the Transformer.
        Depending on the skips, some operations can be skipped.
        If out_computation=True, the output dictionary contains every vectors computed. 
        """

        seq_len = x.shape[1]
        context_window = self.meta_params['context_window']
        assert seq_len <= context_window

        attn_mask = (t.tril(t.ones((seq_len, seq_len))) == 0).to(self.device)

        res = self.word_emb.weight[x]
        pos = self.pos_emb.weight[:seq_len].unsqueeze(0)

        for para_attn, mlp in zip(self.attn_seq, self.mlp_seq):
            if self.meta_params['d'] != self.meta_params['d_head']:
                compressed_res = self.linear_compress(res)
            else:
                compressed_res = res
            para_res = t.zeros_like(compressed_res)
            
            # Each attention mechanism is computed in parallel and then added in the stream.
            for attn in para_attn: 
                attn_j, _ = attn(
                    (compressed_res+pos), 
                    (compressed_res+pos), 
                    compressed_res+pos, 
                    attn_mask=attn_mask
                )
                para_res += attn_j

            if self.meta_params['d'] != self.meta_params['d_head']:
                decompressed_res = self.linear_decompress(para_res)
            else:
                decompressed_res = para_res

            res = decompressed_res + res
                
            norm_res = layer_norm(res)
            if self.use_mlp:
                mlp_out = mlp(norm_res)
            else:
                mlp_out = t.zeros_like(norm_res)
            res = mlp_out + res
            
        logits: t.Tensor = self.unemb(res) #no layer-norm at the end, we want modular temperature
        logits = logits - logits.mean()
        return logits
    

class AoT(Transformer):
    def __init__(self, d: int, N: int, nb_layers: int, parallel_heads: int, d_head: int, nb_head: int, context_window: int, pi: List[t.Tensor]) -> None:
        super().__init__(d, N, nb_layers, 0, 0, parallel_heads, d_head, nb_head, context_window, pi)


class Low_rank(t.nn.Module):
    def __init__(self, d: int, N: int, context_window: int, pi: List[t.Tensor], device: str = 'cpu') -> None:
        super().__init__()
        n_gram = len(pi)
        self.word_emb = t.nn.Linear(d, N**(n_gram-1), bias=False)
        self.unemb = t.nn.Linear(d, N, bias=False)

        self.meta_params: Dict[str, int] = {
            'd': d,
            'N': N,
            'nb_layers': 0,
            'width': 0,
            'depth': 0,
            'para': 0,
            'd_head': d,
            'nb_head': 0,
            'context_window': context_window,
            'n_gram': n_gram,
        }
        self.pi: List[t.Tensor] = pi
        self.device: str = device

    def forward(self, x: t.Tensor) -> t.Tensor: #works for trigram only
        x = x[:, :-1] + x[:, 1:]*self.meta_params['N']

        #concatenates anything in first position since we don't care about i-th prediction for i < n-gram - 1
        x = t.cat([t.zeros(x.shape[0], 1).to(t.int).to(self.device), x], dim=1) 

        logits = self.unemb(self.word_emb.weight[x])
        logits = logits - logits.mean()
        return logits