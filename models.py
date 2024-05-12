import torch as t
from typing import Dict, List, Tuple
from utils import layer_norm

device = 'cpu' #mps is way slower!
cosim = t.nn.CosineSimilarity(dim=-1)
default_skips: Dict[str, bool] = {
    'skip_res_connection': False,
    'skip_pos_QK': False,
    'skip_emb_QK': False,
    'skip_pos_OV': False,
    'skip_emb_OV': False,
}


class Transformer(t.nn.Module):
    """Transformer architecture with parallel attention heads and MLPs, additive positional embeddings, and layer-norm."""
   
    def __init__(
            self, d: int, N: int, nb_layers: int, width: int, depth: int, 
            parallel_heads: int, nb_head: int, context_window: int, 
            pi: List[t.Tensor], skips: Dict[str, bool] = default_skips
            ) -> None:
        """
        Parameters.
        d: embedding dimension,
        N: size of the vocabulary,
        width: width of the MLP,
        depth: depth of the MLP,
        nb_head: number of sub-heads in an attention module, it should divide d,
        nb_layers: number of layers in the Transformer,
        context_window: maximum length of a sequence of tokens,
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
            'nb_head': nb_head,
            'context_window': context_window,
            'n_gram': len(pi),
        }
        self.skips: Dict[str, bool] = skips
        self.pi: List[t.Tensor] = pi

        super().__init__()

        self.word_emb = t.nn.Linear(d, N, bias=False)
        self.pos_emb = t.nn.Linear(d, context_window, bias=False) #Additive positional embedding

        #Implements para attention module in parallel at head layers, each containing nb_head heads.
        self.attn_seq = t.nn.Sequential(
            *[
                t.nn.Sequential(
                    *[t.nn.MultiheadAttention(d, nb_head, batch_first=True, bias=False) for _ in range(parallel_heads)]
                )
            for _ in range(nb_layers)]
        )

        #Implements MLPs with fixed width and depth at each layer.
        #Doesn't implements if 0 layers of 0 hidden layers or 0 width.
        if self.use_mlp:
            self.mlp_seq = t.nn.Sequential(
                *[t.nn.Sequential(
                    *([t.nn.Linear(d, width, bias=True)] + [t.nn.GELU() if i%2 == 0 else t.nn.Linear(width, width, bias=True) for i in range(2*(depth-1)+1)] + [t.nn.Linear(width, d, bias=False)])
                ) for _ in range(nb_layers)]
            )
        else:
            self.mlp_seq = t.nn.Sequential(*[t.nn.Sequential()]*len(self.attn_seq))

        self.unemb = t.nn.Linear(d, N, bias=False)


    def freeze(self, freezer: Dict[str, bool]) -> None:
        freeze_E = not(freezer['freeze_E'])
        freeze_pos = not(freezer['freeze_pos'])
        freeze_QKV = not(freezer['freeze_QKV'])
        freeze_O = not(freezer['freeze_O'])
        freeze_U = not(freezer['freeze_U'])

        self.word_emb.requires_grad_(freeze_E)
        self.pos_emb.requires_grad_(freeze_pos)

        for para_attn in self.attn_seq:
            attn: t.nn.MultiheadAttention
            for attn in para_attn:
                attn.in_proj_weight.requires_grad_(freeze_QKV)
                attn.out_proj.requires_grad_(freeze_O)

        self.unemb.requires_grad_(freeze_U)


    def forward(self, x: t.Tensor, out_computation=False) -> Tuple[t.Tensor, Dict[str, t.Tensor]]:
        """
        Computes the forward pass of the Transformer.
        Depending on the skips, some operations can be skipped.
        If out_computation=True, the output dictionary contains every vectors computed. 
        """

        seq_len = x.shape[1]
        context_window = self.meta_params['context_window']
        assert seq_len <= context_window

        attn_mask = (t.tril(t.ones((seq_len, seq_len))) == 0).to(device)
        computation: Dict[str, t.Tensor] = {}

        #We look at possible computation short-cut.
        skips = self.skips
        skip_res_connection = 0 if skips['skip_res_connection'] else 1
        skip_pos_QK = 0 if skips['skip_pos_QK'] else 1
        skip_emb_QK = 0 if skips['skip_emb_QK'] else 1
        skip_pos_OV = 0 if skips['skip_pos_OV'] else 1
        skip_emb_OV = 0 if skips['skip_emb_OV'] else 1

        res = self.word_emb.weight[x]
        pos = self.pos_emb.weight[:seq_len].unsqueeze(0)
        if out_computation:
            computation[f'res_{0}'] = res
            computation[f'pos'] = pos

        for i, (para_attn, mlp) in enumerate(zip(self.attn_seq, self.mlp_seq)):
            norm_res = layer_norm(res) #we add the positional embedding at each layer to make it more efficient
            para_res = t.zeros_like(res)

            for j, attn in enumerate(para_attn): #if there is parallel attention, each mechanism is computed in parallel and then added in the stream
                attn_j, _ = attn(
                    norm_res*skip_emb_QK+pos*skip_pos_QK, 
                    norm_res*skip_emb_QK+pos*skip_pos_QK, 
                    norm_res*skip_emb_OV+pos*skip_pos_OV, 
                    attn_mask=attn_mask
                )

                para_res += attn_j
                if out_computation:
                    computation[f'para_{j}_layer_{i}'] = attn_j

            res = para_res + res*skip_res_connection
            if out_computation:
                computation[f'res_after_attn_layer_{i}'] = res
                
            norm_res = layer_norm(res)
            if self.use_mlp:
                mlp_out = mlp(norm_res)
            else:
                mlp_out = t.zeros_like(norm_res)
            res = mlp_out + res
            if out_computation:
                computation[f'mlp_{i}'] = mlp_out
                computation[f'res_after_mlp_layer_{i}'] = res
            
        logits: t.Tensor = self.unemb(res) #no layer-norm at the end, we want modular temperature
        logits = logits - logits.mean()
        if out_computation:
            computation[f'logits'] = logits
        return logits, computation
    

class Low_rank(t.nn.Module):
    def __init__(self, d: int, N: int, context_window: int, pi: List[t.Tensor]) -> None:
        super().__init__()
        n_gram = len(pi)
        self.word_emb = t.nn.Linear(d, N**(n_gram-1), bias=False)
        self.unemb = t.nn.Linear(d, N, bias=False)

        self.meta_params: Dict[str, int] = {
            'd': d,
            'N': N,
            'width': 0,
            'depth': 0,
            'nb_head': 0,
            'context_window': context_window,
            'nb_layers': 0,
            'para': 0,
            'n_gram': n_gram,
        }
        self.pi: List[t.Tensor] = pi


    def compute_div(self):
        """
        Compute the closed-form divergence.
        """
        with t.no_grad():
            W_E = self.word_emb.weight.detach()
            W_U = self.unemb.weight.detach()
            f = W_E@W_U.mH 
            L = t.log(self.pi[2].flatten(0, 1))
            PI = self.pi[2].flatten(0, 1)
            Z = f - L - ((f-L)*PI).sum(-1, keepdim=True)
            return t.log((t.exp(Z)*PI).sum(-1)).mean()


    def freeze(self, freezer: Dict[str, bool]) -> None:
        """
        Freezes the training of the embedding and/or unembedding.
        """
        freeze_E = not(freezer['freeze_E'])
        freeze_U = not(freezer['freeze_U'])

        self.word_emb.requires_grad_(freeze_E)
        self.unemb.requires_grad_(freeze_U)


    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, Dict[str, t.Tensor]]: #works for trigram only
        x = x[:, :-1] + x[:, 1:]*self.meta_params['N']

        #concatenates anything in first position since we don't care about i-th prediction for i < n-gram - 1
        x = t.cat([t.zeros(x.shape[0], 1).to(t.int).to(device), x], dim=1) 

        logits = self.unemb(self.word_emb.weight[x])
        logits = logits - logits.mean()
        return logits, {}