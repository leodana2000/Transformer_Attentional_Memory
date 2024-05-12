import torch as t
import numpy as np
import seaborn as sns
from typing import List, Dict
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


def new_computation_basis(model: Transformer, examples: t.Tensor) -> Dict['str', t.Tensor]:
    """
    Changes the computations into a new basis.
    """
    with t.no_grad():
        _, computation = model.forward(examples, out_computation=True)

    W_U = model.unemb.weight.detach()
    pseudo_inv_W_U = W_U@t.linalg.inv(W_U.mH@W_U)

    new_computations = {}
    for key in computation.keys():
        if key != 'logits':
            new_computations[key] = computation[key]@(pseudo_inv_W_U.mH)
        else:
            new_computations[key] = computation[key]

    return new_computations


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
        paras = model.meta_params['para']

        mean_logit = computation['logits'][:, seq_target-1].mean(dim=-1)

        contribution_list = []
        for id in range(N):
            direction = model.unemb.weight[id]

            id_list = []
            for layer in range(nb_layers):
                layer_list = []
                for para in range(paras):
                    layer_list.append(np.array(t.einsum('d, ...d -> ...', direction, computation[f'para_{para}_layer_{layer}'][:, seq_target-1]).detach()-mean_logit))
                id_list.append(layer_list)

                id_list.append([np.array(t.einsum('d, ...d -> ...', direction, computation[f'res_{layer}'][:, seq_target-1]).detach()-mean_logit)])
                id_list.append([np.array(t.einsum('d, ...d -> ...', direction, computation[f'res_after_attn_layer_{layer}'][:, seq_target-1]).detach()-mean_logit)])
            contribution_list.append(id_list)

    return contribution_list

    
def by_attention(contribution: List[List[List[np.ndarray]]], examples: t.Tensor, layer: int, para: int, seq_target: int = 2, sort = False, method = 'solo') -> t.Tensor:
    """
    Project for an attention module, its contribution on each correct next_token, 
    and compares it with the max logit contribution of this head, and other heads.
    """
    N = len(contribution)
    next_tokens = examples[:, seq_target]
    nb_tokens = len(next_tokens)
    
    if method == 'group':
        contrib = t.tensor(np.array([contrib[-1] for contrib in contribution]))
        contrib_l_para = contrib[next_tokens[t.arange(nb_tokens)], layer, t.arange(nb_tokens)]
        contrib_max = t.concat([t.max(contrib[t.logical_not(next_tokens[j] == t.arange(N))][:, layer, j], dim=0)[0].unsqueeze(0) for j in t.arange(nb_tokens)], dim=0)
    elif method == 'solo':
        contrib = t.tensor(np.array([contrib[:-2] for contrib in contribution]))
        contrib_l_para = contrib[next_tokens[t.arange(nb_tokens)], layer, para, t.arange(nb_tokens)]
        contrib_max = t.concat([t.max(contrib[t.logical_not(next_tokens[j] == t.arange(N))][:, layer, para, j], dim=0)[0].unsqueeze(0) for j in t.arange(nb_tokens)], dim=0)
    elif method == '':
        pass
    token_num = t.arange(len(contrib_l_para))

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


def every_attention(contribution: List[List[List[np.ndarray]]], examples: t.Tensor, layer: int, seq_target: int = 2) -> None:
    """
    """
    N = len(contribution)
    next_token = examples[0, seq_target].to(t.int).item()
    nb_heads = len(contribution[0][0])

    contrib = t.tensor(np.array([contrib[:-2] for contrib in contribution]))
    contrib_l_para = contrib[:, layer, :, 0]
    contrib_max = t.max(contrib_l_para[t.logical_not(t.arange(N) == next_token)], dim = 0, keepdim=True)[0]

    contrib = (contrib_l_para-contrib_max).unsqueeze(-1)
    contrib = t.concat([contrib[:, i] for i in range(nb_heads)] + [contrib.sum(dim=1)], dim = 1)
    best = t.argmax(contrib.sum(dim=1))

    sns.heatmap(contrib, center=0, xticklabels = [f"Head {i}" for i in range(nb_heads)]+["Both"], yticklabels=["Correct and best" if i==next_token and i==best else "Best" if i==best else "Correct" if i==next_token else f"token {i}" for i in range(N)], cmap='bwr')
    plt.ylabel('Output token')
    plt.title("Differential contribution to the correct logit for each possible pair.")
    plt.show()


#SVD of W_O
def gather_W_O(model: Transformer):
    W_O_list = []
    for attn in model.attn_seq:
        att: t.nn.MultiheadAttention
        for att in attn:
            W_O_list.append(att.out_proj.weight)
    W_O = t.concat(W_O_list, dim=0)
    return W_O