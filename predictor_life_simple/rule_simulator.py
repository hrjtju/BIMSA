from collections import Counter
from itertools import product
import os
from typing import Callable, Dict, List, Mapping, Optional
from functools import partial, reduce
from operator import add

from matplotlib import pyplot as plt
import torch
from torch import nn, Tensor
import einops
from tqdm import tqdm

import seagull as sgl
from seagull.rules import life_rule

from life1 import (life_rule_monkey_patch, board_init_monkey_patch, simulator_run_monkey_patch, _count_neighbors_monkey_patch, _parse_rulestring_monkey_patch)

life_rule = life_rule_monkey_patch
sgl.Board.__init__ = board_init_monkey_patch
sgl.Simulator.run = simulator_run_monkey_patch

def stat_fn(l: List[List[int]]) -> List[List[int]]:
    
    d = [[-1], [-1], [-1], [-1]]
    
    for (i, c, o) in tqdm(l):
        k = int(2*i + o)
        d[k].append(c)
        
    return d

class CountingCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.counting_kenel = torch.nn.Conv2d(1, 1, 3, 1, 1, padding_mode="circular", bias=False)
        
        with torch.no_grad():
            self.counting_kenel.weight.copy_(torch.tensor(
                [[[
                    [1,  1,  1],
                    [1,  0,  1],
                    [1,  1,  1]
                ]]],
                requires_grad=False
            ))
        
        for param in self.parameters():
            param.requires_grad_(False)
    
    def forward(self, x):
        return self.counting_kenel(x)

class RuleSimulatorStats:
    def __init__(self, 
                 rule: str = None, 
                 ):
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        
        self.count_ker = CountingCNN().to(self.device)
        self.rule = rule.replace('_', '/')
        self.output = None
        
        self.input = self.construct_input()
    
    def load_model(self, model: nn.Module, 
                 param_file: Optional[str] = None):
        
        self.model: Callable[[torch.Tensor], torch.Tensor] = model.to(self.device)
        
        if param_file is not None:
            try:
                self.rule = param_file.split('-')[-1].replace('_', '/')
                self.model.load_state_dict(torch.load(param_file, map_location=self.device))
            except RuntimeError:
                print("Loading params failed.")
    
    def construct_input(self):
        seq_len = 50
        data_ls = []
        
        for _ in range(40):
            board = sgl.Board((200, 200), p_pos=0.5)
            rule = partial(life_rule, rulestring=self.rule)
        
            sim = sgl.Simulator(board)
            sim.run(rule, seq_len)
            
            data_ls.append(torch.tensor(sim.get_history(exclude_init=True)).float())
        
        return torch.concat(data_ls, dim=0)
    
    @torch.no_grad()
    def _predict(self):
        return self.model(self.input).argmax(dim=1, keepdim=True)
    
    @torch.no_grad()
    def _count(self):
        return self.count_ker(self.input)
    
    def get_transform_stats(self, iter_idx: int = 0):
        self.iter_idx = iter_idx
        
        out = self._predict()
        count = self._count()
        
        stats_arr = torch.stack([self.input, count, out], dim=0)
        stats_arr = list(einops.rearrange(stats_arr, "n b c w h -> (b c w h) n").cpu().numpy())
        
        return stat_fn(stats_arr)

    def plot_transform_stats(self, stat_ls, out_dir):
        
        titles = ["dead $\\rightarrow$ dead/living",
                  "living $\\rightarrow$ dead/living"]
        counters = [(Counter(i[1:])) for i in stat_ls]
        stats = [(c.keys(), c.values()) for c in counters]
        
        plt.figure(dpi=200, figsize=(8, 4))

        plt.suptitle(f"Stats of neural network predicted dynamics w.r.t. rule {self.rule}")

        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.bar(*stats[2*i], width=0.4, align="center", label="$\\rightarrow$ dead", alpha=0.5)
            plt.bar(*stats[2*i+1], width=0.4, align="center", label="$\\rightarrow$ living", alpha=0.5)
            plt.xticks(range(9), range(9))
            plt.semilogy()
            plt.grid()
            plt.legend()
            plt.title(titles[i])

        plt.tight_layout(pad=1, h_pad=0.5, w_pad=0.5)
        
        if out_dir is not None:
            plt.savefig(os.path.join(out_dir, f"stats_out-{self.rule.replace('/', '_')}-{self.iter_idx:>04}.svg"), format="svg")

listmap = lambda f,l: list(map(f,l))

class RuleSimulatorDict:
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

if __name__ == "__main__":
    from model_conv import SimpleP4CNNSmalL2Layer
    import e2cnn
    
    model = SimpleP4CNNSmalL2Layer()
    param_file = r"D:\Internship\bimsa\result\predictor_life_simple\2025-11-24_13-42-59_AL_small_2_layer_seq_p4cnn__200-200-B35678_S5678\round_02\best_simple_life_SimpleP4CNNSmall_0.1.0-p4.pth"
    
    simulator = RuleSimulatorStats(model, param_file)
    
    stat_ls = simulator.get_transform_stats()
    
    simulator.plot_transform_stats(stat_ls, "./")