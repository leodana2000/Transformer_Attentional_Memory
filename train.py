import torch as t
from tqdm import tqdm
from typing import Dict, List, Union
from torch.utils.data import DataLoader
from utils import entropy
from models import Transformer, Low_rank


device = 'cpu' #mps is way slower!


def compute_loss(model: Union[Transformer, Low_rank], batch: t.Tensor, ent: t.Tensor, loss_fn, next_token: bool) -> t.Tensor:
    """
    Computes the loss of the model on a batch, and adds the entropy for normalization.
    If next_token=True, the predictions are compared with the next token.
    Otherwise, the predictions are compared with the full probability from pi.
    """

    batch = batch.to(device)
    model_logits = model(batch)[0]
    model_proba = t.softmax(model_logits, dim=-1)
    pred_proba = model_proba[:, 1:-1, :]+1e-12

    if next_token:
        target = batch[:, 2:]
        loss = - ent + loss_fn(t.log(pred_proba.flatten(0, 1)), target.flatten(0, 1))
    else:
        true_proba = model.pi[2][batch[:, :-2], batch[:, 1:-1]].detach()
        loss = - ent - (true_proba*t.log(pred_proba)).sum(-1).mean()
    del batch
    return loss


def compute_acc(model: Union[Transformer, Low_rank], batch: t.Tensor) -> t.Tensor:
    """
    Compute the accuracy of the model for predicting the correct token.
    Meaningfull only if the task is to learn a look-up table, low-entropy distribution.
    """
    with t.no_grad():
        model_logits = model(batch)[0]
        predictions = t.argmax(model_logits, dim=-1)
        target = batch[:, 2:]
        acc = (predictions[:, 1:-1] == target).to(t.float).mean()
    return acc


def train(model: Union[Transformer, Low_rank], dataloader: DataLoader, lr: float=1e-3, next_token: bool=True, seed: int=0) -> Dict[str, List[float]]:
    """
    Trains the model and return the loss and accuracy over all batches.
    """
    
    t.manual_seed(seed)
    model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    ent = entropy(model.pi).to(device)
    loss_fn = t.nn.CrossEntropyLoss()

    Loss = []
    Acc = []
    for batch in tqdm(dataloader):
        loss = compute_loss(model, batch[0], ent, loss_fn, next_token)
        acc = compute_acc(model, batch[0]) #incorrect
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        Loss.append(loss.item())
        Acc.append(acc.item())
    model.to('cpu')
    
    return {'Loss': Loss, 'Acc': Acc}


def train_boosting(model: Union[Transformer, Low_rank], dataloaders: List[DataLoader], lr: float=1e-3, next_token: bool=True, seed: int=0) -> Dict[str, List[float]]:
    """
    Trains the model head by head, and return the loss and accuracy over all batches.
    """

    para = model.meta_params['para']
    freezer = {
        'freeze_E': False,
        'freeze_pos': False,
        'freeze_U': False,
        'freeze_Attn': [[{'freeze_O': True, 'freeze_QKV': True} for _ in range(para)]]
    }
    model.freeze(freezer)
    model.skips['skip_attn'] = [[True for _ in range(para)]]

    dict: Dict[str, List[float]] = {'Loss': [], 'Acc': []}
    for para in range(para):
        model.skips['skip_attn'][0][para] = False

        freezer['freeze_Attn'][0][para]['freeze_O'] = False
        freezer['freeze_Attn'][0][para]['freeze_QKV'] = False
        for ind in range(para):
            freezer['freeze_Attn'][0][ind]['freeze_O'] = True
            freezer['freeze_Attn'][0][ind]['freeze_QKV'] = True
        model.freeze(freezer)


        dict_para = train(model, dataloaders[para], lr, next_token=next_token, seed=seed)
        dict['Loss'] += dict_para['Loss']
        dict['Acc'] += dict_para['Acc']

        freezer['freeze_E'] = True
        freezer['freeze_U'] = True
        freezer['freeze_pos'] = True
    
    return dict