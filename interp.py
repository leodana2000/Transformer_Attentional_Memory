import torch as t
import numpy as np
import seaborn as sns
import math
from typing import List, Dict, Tuple
from models import Transformer, layer_norm
import matplotlib.pyplot as plt
import ternary
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable
from utils import generate_uniform_simplex, generate_uniform_sphere, layer_norm
import plotly.graph_objects as go


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

        contribution_list = []
        for id in range(N):
            direction = model.unemb.weight[id]

            id_list = []
            for layer in range(nb_layers):
                layer_list = []
                for para in range(paras):
                    layer_list.append(np.array(t.einsum('d, ...d -> ...', direction, computation[f'para_{para}_layer_{layer}'][:, seq_target-1]).detach()))
                id_list.append(layer_list)

                id_list.append([np.array(t.einsum('d, ...d -> ...', direction, computation[f'res_{layer}'][:, seq_target-1]).detach())])
                id_list.append([np.array(t.einsum('d, ...d -> ...', direction, computation[f'res_after_attn_layer_{layer}'][:, seq_target-1]).detach())])
            contribution_list.append(id_list)

    return contribution_list #Shape: [N, nb_layers, nb_para, batch_size]

    
def by_attention(contribution: List[List[List[np.ndarray]]], examples: t.Tensor, layer: int, para: int, seq_target: int = 2, sort = False, method = 'solo') -> None:
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


def every_attention(contribution: List[List[List[np.ndarray]]], examples: t.Tensor, ex: int, layer: int, seq_target: int = 2) -> None:
    """
    Plots ...
    """

    N = len(contribution)
    next_token = examples[ex, seq_target].to(t.int)
    nb_heads = len(contribution[0][0])

    contrib = t.tensor(np.array([contrib[:-2] for contrib in contribution]))
    contrib_l_para = contrib[:, layer, :, ex]
    contrib_max = contrib_l_para[next_token].unsqueeze(0)

    contrib = (contrib_l_para-contrib_max).unsqueeze(-1)
    contrib = t.concat([contrib[:, i] for i in range(nb_heads)] + [contrib[:, 0]+contrib[:, 1], contrib[:, 0]+contrib[:, 2], contrib[:, 2]+contrib[:, 1]] + [contrib.sum(dim=1)], dim = 1) #We concatenate each head's output and the combination of all heads
    best = t.argmax(contrib.sum(dim=1))

    sns.heatmap(contrib, center=0, xticklabels = [f"Head {i}" for i in range(nb_heads)]+["Head 0&1", "Head 0&2", "Head 1&2"]+["All"], yticklabels=["Correct and best" if i==next_token and i==best else "Best" if i==best else "Correct" if i==next_token else f"token {i}" for i in range(N)], cmap='bwr')
    plt.ylabel('Output token')
    plt.title("Contribution of each head to each logits\n centered at the correct next token.")
    plt.show()


def single_attention(contribution: List[List[List[np.ndarray]]], examples: t.Tensor, para: int, layer: int, seq_target: int = 2) -> None:
    next_token = examples[:, seq_target].to(t.int)

    contrib = t.tensor(np.array([contrib[:-2] for contrib in contribution]))
    contrib_l_para = contrib[:, layer, para, :]
    contrib_max = contrib_l_para[next_token, t.arange(len(next_token))].unsqueeze(0)

    contrib = contrib_l_para-contrib_max

    sns.heatmap(contrib, center=0, cmap='bwr')
    plt.ylabel('Output token')
    plt.title(f"Token pairs output for head {para}.")
    plt.show()


def attention_by_output(contribution: List[List[List[np.ndarray]]], examples: t.Tensor, layer: int, seq_target: int = 2):
    """
    Prints the accuracy the predictors: Head 0, Head 1, Head 2, Heads 0&1, Heads 0&2, Heads 1&2, Heads 0&1&2.
    """

    N = len(contribution)
    next_token = examples[:, seq_target].to(t.int)

    contributions = t.tensor(np.array([contrib[:-2] for contrib in contribution]))

    contrib_l_para = contributions[:, layer]
    contrib_max = t.cat([contrib_l_para[next_token[ind], :, ind].unsqueeze(0) for ind in range(N)], dim=0)

    contrib = (contrib_l_para-contrib_max).unsqueeze(-1)
    nice_pred_0 = (t.max(contrib[:, 0], dim=0)[0] <= 0).to(t.float).mean().item()
    nice_pred_1 = (t.max(contrib[:, 1], dim=0)[0] <= 0).to(t.float).mean().item()
    nice_pred_2 = (t.max(contrib[:, 2], dim=0)[0] <= 0).to(t.float).mean().item()
    nice_pred_01 = (t.max(contrib[:, 0]+contrib[:, 1], dim=0)[0] <= 0).to(t.float).mean().item()
    nice_pred_02 = (t.max(contrib[:, 0]+contrib[:, 2], dim=0)[0] <= 0).to(t.float).mean().item()
    nice_pred_12 = (t.max(contrib[:, 1]+contrib[:, 2], dim=0)[0] <= 0).to(t.float).mean().item()
    nice_pred_012 = (t.max(contrib.sum(dim=1), dim=0)[0] <= 0).to(t.float).mean().item()

    print('all', nice_pred_0, nice_pred_1, nice_pred_2, nice_pred_01, nice_pred_02, nice_pred_12, nice_pred_012)



# Head contribution plots

def ternary_plot(contribution: List[List[List[np.ndarray]]], examples: t.Tensor, layer: int, seq_target: int = 2, input=None, output=None, comp_method: str = 'combined', plot_method: str = 'logit') -> None:
    """
    Plots ...
    """
    assert comp_method in ['combined', 'input', 'output']
    assert plot_method in  ['logit', 'proba', 'div', 'accuracy', 'hinge']
    if comp_method == 'input':
        assert input is not None
    elif comp_method == 'output':
        assert output is not None

    N = len(contribution)
    contrib = t.tensor(np.array([contrib[:-2] for contrib in contribution]))

    # Computes the contribution according to the good method.
    if comp_method == 'input':
        select = examples[input, seq_target].to(t.int)
    elif comp_method == 'output':
        select = (examples[:, seq_target] == output)
    elif  comp_method == 'combined':
        select = t.arange(len(examples))

    next_tokens = examples[:, seq_target].to(t.int)
    nb_tokens = len(next_tokens)
    contrib_l_para = contrib[:, layer, :, :]
    contrib_max = contrib_l_para[next_tokens, :, t.arange(nb_tokens)].mH.unsqueeze(0)
    contrib = contrib_l_para-contrib_max


    # Initialize a ternary plot
    _, ax = plt.subplots(figsize=(10, 10))
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=1.)
    if comp_method == 'input':
        tax.set_title(f"Ternary Token {input}", fontsize=20)
    elif comp_method == 'output':
        tax.set_title(f"Ternary Output Token {output}", fontsize=20)
    elif comp_method == 'combined':
        tax.set_title("Ternary Combine Token", fontsize=20)

    # Set corner labels
    tax.right_corner_label("Head 0", fontsize=15)
    tax.top_corner_label("Head 1", fontsize=15)
    tax.left_corner_label("Head 2", fontsize=15)

    # Sample data points (p, q, r) on the simplex
    simplex_coordinate = generate_uniform_simplex(nb_points=5000, add_special=True)
    simplex_value = []
    for p, q, r in simplex_coordinate:
        mixture = contrib[:, 0]*p + contrib[:, 1]*q + contrib[:, 2]*r

        if plot_method == 'logit':
            max_logit = t.Tensor([t.max(mixture[t.arange(N) != next_tokens[j], j], dim=0)[0].item() for j in t.arange(nb_tokens)])[select].mean().item()
            simplex_value.append(max_logit)

        elif plot_method == 'proba':
            proba = t.softmax(mixture, dim=0)[next_tokens, t.arange(nb_tokens)][select].mean().item()
            simplex_value.append(proba)

        elif plot_method == 'div':
            div = -t.log(1e-10+t.softmax(mixture, dim=0)[next_tokens, t.arange(nb_tokens)])[select].mean().item()
            simplex_value.append(div)

        elif plot_method == 'accuracy':
            acc = (t.Tensor([t.max(mixture[t.arange(N) != next_tokens[j], j], dim=0)[0].item() for j in t.arange(nb_tokens)]) <= 0).to(t.float)[select].mean().item()
            simplex_value.append(acc)

        elif plot_method == 'hinge':
            hinge_loss = t.Tensor([max(t.max(mixture[t.arange(N) != next_tokens[j], j], dim=0)[0].item(), 0) for j in t.arange(nb_tokens)])[select].mean().item()
            simplex_value.append(hinge_loss)


    # Initialize Colormaps and compute each points' color.
    norm: Normalize | TwoSlopeNorm
    if plot_method == 'logit':
        c = 'bwr'
        c_label = 'Logits'
        norm = TwoSlopeNorm(vmin=min(simplex_value), vcenter=0, vmax=max(simplex_value))
    elif plot_method == 'proba':
        c = 'Greens'
        c_label = 'Probability'
        norm = Normalize(vmin=min(simplex_value), vmax=max(simplex_value))
    elif plot_method == 'div':
        c = 'Purples'
        c_label = 'Divergence'
        norm = Normalize(vmin=min(simplex_value), vmax=max(simplex_value))
    elif plot_method == 'accuracy':
        c = 'Oranges'
        c_label = 'Accuracy'
        norm = Normalize(vmin=min(simplex_value), vmax=max(simplex_value))
    elif plot_method == 'hinge':
        c = 'Blues'
        c_label = 'Hinge loss'
        norm = Normalize(vmin=min(simplex_value), vmax=max(simplex_value))
    cmap = plt.get_cmap(c)
    colors = cmap(norm(simplex_value))

    # Plot data points with colors
    for (point, color) in zip(simplex_coordinate, colors):
        tax.scatter([point], marker='o', color=color)

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(c_label, fontsize=15)
        
    # Set ticks and gridlines
    tax.gridlines(color="black", multiple=0.1)
    tax.ticks(axis='lbr', linewidth=1, multiple=0.1, tick_formats="")
    tax.clear_matplotlib_ticks()

    # Show the plot
    plt.show()


def sphere_plot(contribution: List[List[List[np.ndarray]]], examples: t.Tensor, layer: int, seq_target: int = 2, input=None, output=None, comp_method: str = 'combined', plot_method: str = 'logit') -> None:
    """
    Plots ...
    """
    assert comp_method in ['combined', 'input', 'output']
    assert plot_method in  ['logit', 'proba', 'div', 'accuracy', 'hinge']
    if comp_method == 'input':
        assert input is not None
    elif comp_method == 'output':
        assert output is not None

    N = len(contribution)
    contrib = t.tensor(np.array([contrib[:-2] for contrib in contribution]))

    # Computes the contribution according to the good method.
    if comp_method == 'input':
        select = examples[input, seq_target].to(t.int)
    elif comp_method == 'output':
        select = (examples[:, seq_target] == output)
    elif  comp_method == 'combined':
        select = t.arange(len(examples))

    next_tokens = examples[:, seq_target].to(t.int)
    nb_tokens = len(next_tokens)
    contrib_l_para = contrib[:, layer, :, :]
    contrib_max = contrib_l_para[next_tokens, :, t.arange(nb_tokens)].mH.unsqueeze(0)
    contrib = contrib_l_para-contrib_max


    # Sample data points (p, q, r) on the simplex
    sphere_coordinate = generate_uniform_sphere(nb_points=10000)
    sphere_value = []
    for p, q, r in zip(*sphere_coordinate):
        mixture = (contrib[:, 0]*p + contrib[:, 1]*q + contrib[:, 2]*r)/np.sqrt(3) 

        if plot_method == 'logit':
            max_logit = t.Tensor([t.max(mixture[t.arange(N) != next_tokens[j], j], dim=0)[0].item() for j in t.arange(nb_tokens)])[select].mean().item()
            sphere_value.append(max_logit)

        elif plot_method == 'proba':
            proba = t.softmax(mixture, dim=0)[next_tokens, t.arange(nb_tokens)][select].mean().item()
            sphere_value.append(proba)

        elif plot_method == 'div':
            div = -t.log(1e-10+t.softmax(mixture, dim=0)[next_tokens, t.arange(nb_tokens)])[select].mean().item()
            sphere_value.append(div)

        elif plot_method == 'accuracy':
            acc = (t.Tensor([t.max(mixture[t.arange(N) != next_tokens[j], j], dim=0)[0].item() for j in t.arange(nb_tokens)]) <= 0).to(t.float)[select].mean().item()
            sphere_value.append(acc)

        elif plot_method == 'hinge':
            hinge_loss = t.Tensor([max(t.max(mixture[t.arange(N) != next_tokens[j], j], dim=0)[0].item(), 0) for j in t.arange(nb_tokens)])[select].mean().item()
            sphere_value.append(hinge_loss)


    # Initialize Colormaps and compute each points' color.
    norm: Normalize | TwoSlopeNorm
    if plot_method == 'logit':
        c = 'bwr'
        c_label = 'Logits'
        norm = TwoSlopeNorm(vmin=min(sphere_value), vcenter=0, vmax=max(sphere_value))
    elif plot_method == 'proba':
        c = 'Greens'
        c_label = 'Probability'
        norm = Normalize(vmin=min(sphere_value), vmax=max(sphere_value))
    elif plot_method == 'div':
        c = 'Purples'
        c_label = 'Divergence'
        norm = Normalize(vmin=min(sphere_value), vmax=max(sphere_value))
    elif plot_method == 'accuracy':
        c = 'Oranges'
        c_label = 'Accuracy'
        norm = Normalize(vmin=min(sphere_value), vmax=max(sphere_value))
    elif plot_method == 'hinge':
        c = 'Blues'
        c_label = 'Hinge loss'
        norm = Normalize(vmin=min(sphere_value), vmax=max(sphere_value))
    cmap = plt.get_cmap(c)
    colors = cmap(norm(sphere_value))
    
    # Convert colors to a Plotly-compatible format
    compatible_colors = [f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})' for r, g, b, a in colors]

    # Select title
    if comp_method == 'input':
        title = f"Ternary Token {input}"
    elif comp_method == 'output':
        title = f"Ternary Output Token {output}"
    elif comp_method == 'combined':
        title = "Ternary Combine Token"


    # Create a scatter plot on the sphere
    fig = go.Figure(data=[go.Scatter3d(
        x=sphere_coordinate[0], y=sphere_coordinate[1], z=sphere_coordinate[2],
        mode='markers',
        marker=dict(
            size=5,
            color=compatible_colors,
            colorbar=dict(title=c_label),
            opacity=1
        )
    )])

    # Update layout for better visualization
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Head 0',
            yaxis_title='Head 1',
            zaxis_title='Head 2',
            aspectmode='data'
        )
    )

    # Show plot
    fig.show()


def colored(ind: int) -> Tuple[str, str]:
    if ind == 0:
        c = 'Greens'
        c2 = 'green'
    elif ind == 1:
        c = 'Reds'
        c2 = 'red'
    elif ind == 2:
        c = 'Blues'
        c2 = 'blue'
    elif ind == 3:
        c = 'Purples'
        c2 = 'purple'
    elif ind == 4:
        c = 'Oranges'
        c2 = 'orange'

    return (c, c2)


def attention_torus(model: Transformer, head: int, nb_points: int=100, wind_down: float=-np.pi, wind_up: float=np.pi):
    """Plots the probability torus of the tokens at position 0, for a given token at position 1. Works only for embedding dimension 3."""
    # Generate points on a torus
    theta = np.linspace(wind_down, wind_up, nb_points)
    phi = np.linspace(wind_down, wind_up, nb_points)
    theta, phi = np.meshgrid(theta, phi)
    theta, phi = theta.flatten(), phi.flatten()

    # Prepare the data
    x1 = np.cos(theta)
    y1 = np.sin(theta)
    x2 = np.cos(phi)
    y2 = np.sin(phi)

    # Each data point should be of size (seq_length, embedding_dim)
    data = np.stack([np.stack([x1, y1, np.zeros_like(x1)], axis=-1), 
                    np.stack([x2, y2, np.zeros_like(x2)], axis=-1)], axis=1)

    # Convert to tensor
    data = t.tensor(data, dtype=t.float32)

    # Layer normalization
    data_normed = layer_norm(data)
    data_positionned = data_normed + model.pos_emb.weight.detach()[:2].unsqueeze(0)

    # Compute attention
    layer = 0
    attn_mask = t.tril(t.ones((2, 2))) == 0
    attn = model.attn_seq[layer][head]
    attention_pattern = attn(data_positionned, data_positionned, data_positionned, attn_mask=attn_mask, need_weights=True, average_attn_weights=True)[1]
    attention_pattern = attention_pattern[:, 1, 0].reshape((nb_points, nb_points)).detach()

    # Prepare for plotting
    _, ax = plt.subplots(1, 1, figsize=(10, 8))
    cmap = plt.get_cmap('Greys')
    norm = Normalize(vmin=0, vmax=1)

    # Plotting with imshow
    im = ax.imshow(attention_pattern, cmap=cmap, norm=norm, extent=(wind_down, wind_up, wind_down, wind_up), origin='lower', aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Probability')

    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(f'Probability of head {head} attending token 1 on a Torus')
    plt.show()


def proba_torus(model: Transformer, nb_points: int=100, wind_down: float=-np.pi, wind_up: float=np.pi, before_W_U: bool=False, maxi: t.Tensor=t.zeros((3)), mini: t.Tensor=t.zeros((3))) -> Tuple[float, float]:
    """Plots the probability torus of the tokens at position 0, for a given token at position 1. Works only for embedding dimension 3."""
    # Generate points on a torus
    theta = np.linspace(wind_down, wind_up, nb_points)
    phi = np.linspace(wind_down, wind_up, nb_points)
    theta, phi = np.meshgrid(theta, phi)
    theta, phi = theta.flatten(), phi.flatten()

    # Prepare the data
    x1 = np.cos(theta)
    y1 = np.sin(theta)
    x2 = np.cos(phi)
    y2 = np.sin(phi)

    # Each data point should be of size (seq_length, embedding_dim)
    data = np.stack([np.stack([x1, y1, np.zeros_like(x1)], axis=-1), 
                    np.stack([x2, y2, np.zeros_like(x2)], axis=-1)], axis=1)

    # Convert to tensor
    data = t.tensor(data, dtype=t.float32)

    # Layer normalization
    data_normed = layer_norm(data)

    # Throught the model
    unembedding_output, computation = model.forward(data_normed, continuous=True, out_computation=True)

    # Transform into probabilities
    softmax_output = t.softmax(unembedding_output, dim=-1)[:, 1]

    N = model.meta_params['N']
    d = model.meta_params['d']

    # Normalize the values
    norm = Normalize(vmin=0, vmax=1)

    # Prepare for plotting
    _, ax = plt.subplots(1, 1, figsize=(10, 8))

    if before_W_U:
        layer=0
        final_embedding = computation[f'res_after_attn_layer_{layer}'].detach()[:, 1].reshape((nb_points, nb_points, d))
        W_U = model.unemb.weight.detach()
        U, S, V = t.linalg.svd(W_U)
        final_embedding = final_embedding@V
        maxi = t.max(t.max(final_embedding.reshape((nb_points*nb_points, d)), dim=0, keepdim=True)[0], maxi.reshape((1, 3))).unsqueeze(0)
        mini = t.min(t.min(final_embedding.reshape((nb_points*nb_points, d)), dim=0, keepdim=True)[0], mini.reshape((1, 3))).unsqueeze(0)
        RGB_embedding = (final_embedding-mini)/(maxi-mini)

        _ = ax.imshow(RGB_embedding.detach(), extent=(wind_down, wind_up, wind_down, wind_up), origin='lower', aspect='auto')

        ax.set_xlabel('Angle Position 1')
        ax.set_ylabel('Angle Position 2')
        ax.set_title('Embedding space on a Torus')

    else:
        combined_colors = np.zeros((nb_points, nb_points, 4))
        max_z = t.max(softmax_output, dim=-1)[0]
        max_z = max_z.reshape(nb_points, nb_points)

        for out_token in range(N):
            z = softmax_output[:, out_token].detach()
            z = z.reshape(nb_points, nb_points)
            mask = (z == max_z)

            cmap = plt.get_cmap(colored(out_token)[0])
            color = cmap(norm(z))

            combined_colors[mask] = color[mask]

        _ = ax.imshow(combined_colors, extent=(wind_down, wind_up, wind_down, wind_up), origin='lower', aspect='auto')

        W_E = layer_norm(model.word_emb.weight.detach())
        for i, v0 in enumerate(W_E):
            for j, v1 in enumerate(W_E):
                # Convert the vectors to angles
                angle_x = t.atan2(v0[1]-v0[2], v0[0]-v0[2])
                angle_y = t.atan2(v1[1]-v1[2], v1[0]-v1[2])

                next_token = t.argmax(model.pi[2][i, j]).item()
                ax.scatter(angle_x, angle_y, color=colored(next_token)[1], label='Additional Points', edgecolor='black')

            
        ax.set_xlabel('Angle Position 1')
        ax.set_ylabel('Angle Position 2')
        ax.set_title('Superposed Probability Outputs on a Torus')
    
    plt.show()

    return maxi, mini



def mapping_torus(model: Transformer, nb_points: int=100, wind_down: float=-np.pi, wind_up: float=np.pi):
    """Plots the probability torus of the tokens at position 0, for a given token at position 1. Works only for embedding dimension 3."""
    # Generate points on a torus
    theta = np.linspace(wind_down, wind_up, nb_points)
    phi = np.linspace(wind_down, wind_up, nb_points)
    theta, phi = np.meshgrid(theta, phi)
    theta, phi = theta.flatten(), phi.flatten()

    # Prepare the data
    x1 = np.cos(theta)
    y1 = np.sin(theta)
    x2 = np.cos(phi)
    y2 = np.sin(phi)

    # Each data point should be of size (seq_length, embedding_dim)
    data = np.stack([np.stack([x1, y1, np.zeros_like(x1)], axis=-1), 
                    np.stack([x2, y2, np.zeros_like(x2)], axis=-1)], axis=1)

    # Convert to tensor
    data = t.tensor(data, dtype=t.float32)

    # Layer normalization
    data_normed = layer_norm(data)

    # Throught the model
    _, computation = model.forward(data_normed, continuous=True, out_computation=True)
    d = model.meta_params['d']
    final_embedding = computation[f'res_after_attn_layer_{0}'].detach()[:, 1].reshape((nb_points, nb_points, d))
    maxi = t.max(final_embedding)
    mini = t.min(final_embedding)

    
    # Apply the function
    xyz = final_embedding#/t.norm(final_embedding, dim=-1, keepdim=True).numpy()
    x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
    xyz_norm = t.norm(xyz, dim=-1).numpy()

    # Create the 3D plot
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, surfacecolor=xyz_norm, colorscale='Reds', showscale=False)])
    fig.update_layout(title='Mapped Torus to Sphere with Dual Color Scales', scene=dict(
        xaxis=dict(nticks=4, range=[mini, maxi]),
        yaxis=dict(nticks=4, range=[mini, maxi]),
        zaxis=dict(nticks=4, range=[mini, maxi])
    ))

    fig.show()