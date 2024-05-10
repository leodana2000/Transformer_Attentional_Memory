import torch as t
from tqdm import tqdm
from typing import Dict, List, Union
from torch.utils.data import DataLoader
from utils import entropy
from models import Transformer, Low_rank


device = 'cpu' #mps is way slower!


def compute_loss(model: Union[Transformer, Low_rank], batch: t.Tensor, ent: t.Tensor, loss_fn, next_token: bool) -> t.Tensor:
    batch = batch.to(device)
    model_logits = model(batch)[0]
    model_proba = t.softmax(model_logits, dim=-1)

    target = batch[:, 2:]
    if next_token:
        loss = - ent + loss_fn(t.log(model_proba[:, 1:-1, :].flatten(0, 1)+1e-12), target.flatten(0, 1))
    else:
        loss = - ent + model.pi[2][batch[:-2], batch[1:-1]]*t.log(model_proba[:, 1:-1, :]+1e-12)
    del batch
    return loss


def compute_acc(model: Union[Transformer, Low_rank], batch: t.Tensor) -> t.Tensor:
    with t.no_grad():
        model_logits = model(batch)[0]
        predictions = t.argmax(model_logits, dim=-1)
        target = batch[:, 2:]
        acc = (predictions == target).to(t.float).mean()
    return acc


def train(model: Union[Transformer, Low_rank], dataloader: DataLoader, lr: float=1e-3, next_token: bool=True, seed: int=0) -> Dict[str, List[float]]:
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