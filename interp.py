import torch as t
import numpy as np
from typing import List
from models import Transformer
import matplotlib.pyplot as plt


#Attention map
def attention_map(model: Transformer, layer: int, para: int):
    skips = model.skips
    skip_pos_QK = 0 if skips['skip_pos_QK'] else 1
    skip_emb_QK = 0 if skips['skip_emb_QK'] else 1

    W_E = model.word_emb.weight
    _, d = W_E.shape
    pos_1 = model.pos_emb.weight[0].unsqueeze(0)
    pos_2 = model.pos_emb.weight[1].unsqueeze(0)
    att = model.attn_seq[layer][para]
    W_Q, W_K, _ = att.in_proj_weight.split(d, dim=0)

    product_1 = W_E*skip_emb_QK+pos_1*skip_pos_QK
    product_2 = W_E*skip_emb_QK+pos_2*skip_pos_QK

    map_1 = t.einsum('Nd, dD, eD, ne -> Nn', product_2, W_Q, W_K, product_1)/np.sqrt(d)
    map_2 = t.einsum('Nd, dD, eD, Ne -> N', product_2, W_Q, W_K, product_2).unsqueeze(1)/np.sqrt(d)
    map = (map_1 - map_2).unsqueeze(2)
    map = t.softmax(t.concat([map, t.zeros_like(map)], dim=2), dim=2)
    map = map[:, :, 0]
    return map


#Logits backtracking
def back_track(model: Transformer, examples: t.Tensor, seq_target: int = 2) -> List[List[List[np.ndarray]]]:
    """
    Computes the contribution of each head to each logit.
    The contribution of the layer l, parallel module p in the direction id is:
        contribution_list[id][l][p]
    and for the residual stream's contribution, it is:
        contribution_list[id][-1][0]
    """

    with t.no_grad():
        _, computation = model.forward(examples, out_computation=True)

        N = model.meta_params['N']
        nb_layers = model.meta_params['nb_layers']
        para = model.meta_params['para']

        mean_logit = computation['logits'][:, seq_target-1].mean(dim=-1)

        contribution_list = []
        for id in range(N):
            direction = model.unemb.weight[id]

            id_list = []
            for i in range(nb_layers):
                layer_list = []
                for j in range(para):
                    layer_list.append(np.array(t.einsum('d, ...d -> ...', direction, computation[f'attn_{j}_para_{i}'][0][:, seq_target-1]).detach()-mean_logit))
                id_list.append(layer_list)

            id_list.append([np.array(t.einsum('d, ...d -> ...', direction, computation[f'res_{0}'][0, seq_target-1]).detach()-mean_logit)])
            contribution_list.append(id_list)

    return contribution_list

    
def by_attention(contribution: List[List[List[np.ndarray]]], examples: t.Tensor, layer: int, para: int, max_para: List[int], seq_target: int = 2, sort: bool = False) -> t.Tensor:
    """
    Project for an attention module, its contribution on each correct next_token, 
    and compares it with the max logit contribution of this head, and other heads.
    """
    N = len(contribution)
    next_tokens = examples[:, seq_target]
    nb_tokens = len(next_tokens)
    
    contrib = t.tensor(np.array([contrib[:-1] for contrib in contribution]))
    contrib_l_para = contrib[next_tokens[t.arange(nb_tokens)], layer, para, t.arange(nb_tokens)]
    contrib_max = t.concat([t.max(contrib[t.logical_not(next_tokens[j] == t.arange(N))][:, layer, para, j], dim=0)[0].unsqueeze(0) for j in t.arange(nb_tokens)], dim=0)
    token_num = t.arange(len(contrib_l_para))
    
    """argmax = [t.max(contrib[t.logical_not(next_tokens[j] == t.arange(N))][:, layer, para, j], dim=0)[1] for j in t.arange(nb_tokens)]

    contrib_max = []
    for p in max_para:
        contrib_max.append([contrib[argmax[j], layer, p, j] for j in t.arange(nb_tokens)])
    token_num = t.arange(len(contrib_l_para))

    width = .8/(len(max_para)+1)
    mean = 1/2 - width/2
    plt.bar(token_num - mean, contrib_l_para, width=width, label='Contribution to the correct logit')
    for para in range(len(max_para)):
        plt.bar(token_num - mean + 2*mean*((para+1)/(len(max_para)+1)), contrib_max[para], width=width, label=f'Max logit contribution head {para}')
    plt.legend()
    plt.show()"""

    if sort:
        contrib = t.sort(contrib_l_para-contrib_max, descending=True)[0]
    else:
        contrib = contrib_l_para-contrib_max

    plt.bar(token_num, contrib)
    plt.xlabel('Token pair id')
    plt.ylabel('Centered logit')
    plt.title("Differential contribution to the correct logit for each possible pair.")
    plt.show()
    return contrib_l_para


#SVD of W_O
def gather_W_O(model: Transformer):
    W_O_list = []
    for attn in model.attn_seq:
        att: t.nn.MultiheadAttention
        for att in attn:
            W_O_list.append(att.out_proj.weight)
    W_O = t.concat(W_O_list, dim=0)
    return W_O