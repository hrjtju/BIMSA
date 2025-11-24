from itertools import product
from typing import Dict, Mapping
from functools import reduce
from operator import add

import torch
from torch import nn, Tensor
import einops

listmap = lambda f,l: list(map(f,l))

class RuleSimulator:
    def __init__(self, n: int = 3):
        self.n: int = n
        self.indicator_dataset: Tensor = None
        self.rule_d: Dict[Tensor, Tensor] = None
        self.rule_map: Mapping[Tensor, Tensor|int] = None
        self.patch_extractor: nn.Conv2d = None
        
        self.construct_motif()
        self.construct_patch_extractor()
    
    def construct_patch_extractor(self):
        self.patch_extractor = nn.Conv2d(
            in_channels=1, out_channels=self.n*self.n,
            kernel_size=self.n, stride=1,
            padding=1, padding_mode="circular",
            bias=False
        )
        self.patch_extractor.weight.data = einops.rearrange(
            torch.eye(self.n*self.n, self.n*self.n),
            "c (w h) -> c 1 w h", w=self.n, h=self.n
        )
        self.patch_extractor.requires_grad_(False)
    
    def construct_motif(self):
        ls = ([0, 1] for _ in range(self.n*self.n))

        for i in product(*ls):
            arr = torch.tensor(list(i)).reshape(1, 3, 3)
            if self.indicator_dataset is None:
                self.indicator_dataset = arr
            else:
                self.indicator_dataset = torch.concatenate([self.indicator_dataset, arr])
        self.indicator_dataset = self.indicator_dataset[:, None, ...].float()
        
        self.indicator_dataset.requires_grad_(False)

    @torch.no_grad()
    def get_rule_from_nn(self, predictor: nn.Module, epsilon: float = 0.1):
        out_ = torch.softmax(predictor(self.indicator_dataset.to(torch.device("cuda"))), dim=1).cpu()
        stack_arr = torch.cat([self.indicator_dataset, out_[:, 0:1, ...]], dim=1)
        
        filtered_arr = stack_arr[(stack_arr[:, 1, 1, 1] < epsilon) \
            + (stack_arr[:, 1, 1, 1] > 1-epsilon)].transpose(1, 0)
        self.rule_d = {i: j[..., 2, 2] for (i,j) in zip(*filtered_arr)}
        
        self.update_rule_map()

    def update_rule_map(self):
        self.rule_map = lambda x: torch.randint(0, 2, (1,)).item() if x not in self.rule_d else self.rule_d[x]
    
    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        
        patches_arr = einops.rearrange(
            self.patch_extractor(x), "n w h -> w h n")
        return torch.tensor(listmap(self.rule_map, 
                                     reduce(add, [[j.long() for j in i] for i in patches_arr]))).float().reshape(*x.shape)
    
    @torch.no_grad()
    def predict_seq(self, init_state: Tensor, t: int) -> Tensor:
        states = torch.empty((t+1, *init_state.shape), dtype=init_state.dtype)
        states[0] = init_state
        
        for i in range(t):
            next_state = self.predict((x:=states[i].float()) / (dim:=(255 if x.max() > 100 else 1)))
            states[i+1] = next_state
        return states

class RuleSimulatorStatistic:
    def __init__(self):
        ...
    