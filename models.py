import torch as t
import numpy as np
from typing import Dict, List, Tuple
from utils import layer_norm, compute_div

device = 'cpu' #mps is way slower!
cosim = t.nn.CosineSimilarity(dim=-1)


class Transformer(t.nn.Module):
    """Transformer architecture with parallel attention heads and MLPs, additive positional embeddings, and layer-norm."""
   
    def __init__(
            self, d: int, N: int, nb_layers: int, width: int, depth: int, 
            parallel_heads: int, nb_head: int, context_window: int, 
            pi: List[t.Tensor], argmax_mode: bool=False,
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

        self.skips = {
            'skip_res_connection': False,
            'skip_pos_QK': False,
            'skip_emb_QK': False,
            'skip_pos_OV': False,
            'skip_emb_OV': False,
            'skip_attn': [[False for _ in range(parallel_heads)] for _ in range(nb_layers)],
        }
        
        self.pi: List[t.Tensor] = pi
        self.argmax_mode: bool=argmax_mode

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

        for attn_layer in self.attn_seq:
            for attn_para in attn_layer:
                attn_para._qkv_same_embed_dim = False
                attn_para.q_proj_weight = t.nn.Parameter(t.empty((d, d//nb_head)))
                attn_para.k_proj_weight = t.nn.Parameter(t.empty((d, d//nb_head)))
                attn_para.v_proj_weight = t.nn.Parameter(t.empty((d, d//nb_head)))
                t.nn.init.xavier_uniform_(attn_para.q_proj_weight)
                t.nn.init.xavier_uniform_(attn_para.k_proj_weight)
                t.nn.init.xavier_uniform_(attn_para.v_proj_weight)
                attn_para.register_parameter('in_proj_weight', None)

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


    def low_init(self, scale: float=1) -> None:
        self.word_emb.weight = t.nn.Parameter(t.randn_like(self.word_emb.weight)*scale/(np.sqrt(self.word_emb.weight.shape[0]*self.word_emb.weight.shape[1])))
        self.pos_emb.weight = t.nn.Parameter(t.randn_like(self.pos_emb.weight)*scale/(np.sqrt(self.pos_emb.weight.shape[0]*self.pos_emb.weight.shape[1])))
        self.unemb.weight = t.nn.Parameter(t.randn_like(self.unemb.weight)*scale/(np.sqrt(self.unemb.weight.shape[0]*self.unemb.weight.shape[1])))
        for attn in self.attn_seq:
            att: t.nn.MultiheadAttention
            for att in attn:
                att.in_proj_weight.data = t.nn.Parameter(t.randn_like(att.in_proj_weight.data)*scale/(np.sqrt(att.in_proj_weight.data.shape[0]*att.in_proj_weight.data.shape[1])))
                att.out_proj.weight = t.nn.Parameter(t.randn_like(att.out_proj.weight)*scale/(np.sqrt(att.out_proj.weight.shape[0]*att.out_proj.weight.shape[1])))


    def freeze(self, freezer) -> None:
        freeze_E = not(freezer['freeze_E'])
        self.word_emb.requires_grad_(freeze_E)

        freeze_pos = not(freezer['freeze_pos'])
        self.pos_emb.requires_grad_(freeze_pos)

        for layer, para_attn in enumerate(self.attn_seq):
            for para, attn in enumerate(para_attn):
                freeze_Q = not(freezer['freeze_Attn'][layer][para]['freeze_Q'])
                freeze_K = not(freezer['freeze_Attn'][layer][para]['freeze_K'])
                freeze_V = not(freezer['freeze_Attn'][layer][para]['freeze_V'])
                freeze_O = not(freezer['freeze_Attn'][layer][para]['freeze_O'])
                attn.q_proj_weight.requires_grad_(freeze_Q)
                attn.k_proj_weight.requires_grad_(freeze_K)
                attn.v_proj_weight.requires_grad_(freeze_V)
                attn.out_proj.requires_grad_(freeze_O)

        freeze_U = not(freezer['freeze_U'])
        self.unemb.requires_grad_(freeze_U)


    def forward(self, x: t.Tensor, out_computation: bool=False, continuous: bool=False) -> Tuple[t.Tensor, Dict[str, t.Tensor]]:
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

        if self.argmax_mode:
            Lambda = 1000
        else:
            Lambda = 1


        if continuous:
            res = x
        else:
            res = self.word_emb.weight[x]
        pos = self.pos_emb.weight[:seq_len].unsqueeze(0)
        if out_computation:
            computation[f'res_{0}'] = res
            computation[f'pos'] = pos

        for layer, (para_attn, mlp) in enumerate(zip(self.attn_seq, self.mlp_seq)):
            norm_res = layer_norm(res) #we add the positional embedding at each layer to make it more efficient
            para_res = t.zeros_like(res)

            for para, attn in enumerate(para_attn): #if there is parallel attention, each mechanism is computed in parallel and then added in the stream
                if not skips['skip_attn'][layer][para]:
                    attn_j, _ = attn(
                        (norm_res*skip_emb_QK+pos*skip_pos_QK)*Lambda, 
                        (norm_res*skip_emb_QK+pos*skip_pos_QK)*Lambda, 
                        norm_res*skip_emb_OV+pos*skip_pos_OV, 
                        attn_mask=attn_mask
                    )

                    para_res += attn_j
                    if out_computation:
                        computation[f'para_{para}_layer_{layer}'] = attn_j

            res = para_res + res*skip_res_connection
            if out_computation:
                computation[f'res_after_attn_layer_{layer}'] = res
                
            norm_res = layer_norm(res)
            if self.use_mlp:
                mlp_out = mlp(norm_res)
            else:
                mlp_out = t.zeros_like(norm_res)
            res = mlp_out + res
            if out_computation:
                computation[f'mlp_{layer}'] = mlp_out
                computation[f'res_after_mlp_layer_{layer}'] = res
            
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


    def low_init(self, scale: float=1) -> None:
        self.word_emb.weight = t.nn.Parameter(t.randn_like(self.word_emb.weight)*scale/(np.sqrt(self.word_emb.weight.shape[0]*self.word_emb.weight.shape[1])))
        self.unemb.weight = t.nn.Parameter(t.randn_like(self.unemb.weight)*scale/(np.sqrt(self.unemb.weight.shape[0]*self.unemb.weight.shape[1])))


    def compute_div(self):
        """
        Compute the closed-form divergence.
        """
        W_E = self.word_emb.weight.detach()
        W_U = self.unemb.weight.detach()
        pi = self.pi
        return compute_div(W_E, W_U, pi)


    def freeze(self, freezer) -> None:
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