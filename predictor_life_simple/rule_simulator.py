from collections import Counter, OrderedDict
from itertools import chain, combinations, product
import os
import random
import sys
from typing import Callable, Dict, List, Mapping, Optional, Tuple
from functools import partial, reduce
from operator import add

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, Tensor, tensor
import einops
from tqdm import tqdm
import loguru

import seagull as sgl
from seagull.rules import life_rule
from dataloader import get_dataloader

from life1 import (life_rule_monkey_patch, board_init_monkey_patch, simulator_run_monkey_patch, _count_neighbors_monkey_patch, _parse_rulestring_monkey_patch)

life_rule = life_rule_monkey_patch
sgl.Board.__init__ = board_init_monkey_patch
sgl.Simulator.run = simulator_run_monkey_patch

def stat_fn(l: List[List[int]], d) -> List[List[int]]:
    
    for (i, c, o) in l:
        k = int(2*i + o)
        d[k].append(c)
        
    return d

def add_zero_val_key(d: dict) -> dict:
    return {k:d[k] if k in d else 0 for k in range(9)}

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    if len(iterable) == 0:
        return [tuple()]
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))


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
        
        self.d_th, self.l_th = None, None
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        
        self.count_ker = CountingCNN().to(self.device)
        self.rule = rule.replace('_', '/')
        
        self.test_loader = get_dataloader(
            data_dir=f"./predictor_life_simple/datasets/200-200-{self.rule.replace('/', '_')}",
            batch_size=64,
            shuffle=True,
            num_workers=0,
            split='train'
        )
        
        self.last_likelihood = None
    
    def load_model(self, model: nn.Module, 
                 param_file: Optional[str] = None):
        
        self.model: Callable[[torch.Tensor], torch.Tensor] = model
        
        if param_file is not None:
            self.model.load_state_dict(torch.load(param_file, map_location=self.device))

        self.model.to(self.device)
        
    def construct_input(self):
        seq_len = 30
        data_ls = []
        
        for _ in range(20):
            board = sgl.Board((200, 200), p_pos=0.5)
            rule = partial(life_rule, rulestring=self.rule)
        
            sim = sgl.Simulator(board)
            sim.run(rule, seq_len)
            
            data_ls.append(torch.tensor(sim.get_history(exclude_init=True)).float())
        
        return torch.concat(data_ls, dim=0)[:, None].to(self.device)
    
    @torch.no_grad()
    def get_transform_stats(self):
        
        self.test_loader = get_dataloader(
            data_dir=f"./predictor_life_simple/datasets/200-200-{self.rule.replace('/', '_')}",
            batch_size=64,
            shuffle=True,
            num_workers=0,
            split='train'
        )
        
        d = [[-1], [-1], [-1], [-1]]

        for idx, (inputs, _) in tqdm(enumerate(self.test_loader), total=10):
            
            inputs = inputs.to(self.device)
            
            #! perturb inputs, but can make result shifted
            #! eg. B3/S23 -> B1/S12
            # inputs = (inputs + 0.5 * torch.randn_like(inputs)).long().clamp(0, 1).float()
            
            counts = self.count_ker(inputs)
            pred = self.model(inputs).argmax(dim=1, keepdim=True)
            
            stat_arr = torch.stack([inputs, counts, pred], dim=0)
            stat_arr = list(einops.rearrange(stat_arr, "n b c w h -> (b c w h) n").cpu().numpy())
            
            stat_fn(stat_arr, d)
            
            if idx == 10:
                break
        
        return d

    def plot_transform_stats(self, stat_ls, out_dir, iter_idx):
        
        self.iter_idx = iter_idx
        
        titles = ["dead $\\rightarrow$ dead/living",
                  "living $\\rightarrow$ dead/living"]
        counters = [(Counter(i[1:])) for i in stat_ls]
        stats = []
        
        for counter in counters:
            items = sorted(list(counter.items()), key=lambda x:x[0])
            # print(items)
            
            x = list(range(9))
            y = [counter.get(i, 0) for i in x]
            stats.append((x,y))
        
        
        # print(counters, stats, sep="\n", end="\n\n")
        
        plt.figure(dpi=200, figsize=(8, 4))

        plt.suptitle(f"Stats of neural network predicted dynamics w.r.t. rule {self.rule}")

        tmp_ls = [self.d_th, self.l_th]
        
        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.bar(*stats[2*i], width=0.4, align="center", label="$\\rightarrow$ dead", alpha=0.5)
            plt.bar(*stats[2*i+1], width=0.4, align="center", label="$\\rightarrow$ living", alpha=0.5)
            plt.xticks(range(9), range(9))
            plt.xlabel("# Living Neighbors")
            plt.ylabel("Freq.")
            
            if tmp_ls[i] is not None:
                plt.axhline(y=tmp_ls[i], color='r', linestyle='--', label="Rule Acceptance Threshold")
            
            plt.semilogy()
            plt.grid()
            plt.legend()
            plt.title(titles[i])

        plt.tight_layout(pad=1, h_pad=0.5, w_pad=0.5)
        
        if out_dir is not None:
            plt.savefig(os.path.join(out_dir, f"stats_out-{self.rule.replace('/', '_')}-{self.iter_idx:>04}.svg"), format="svg")
        
        plt.close()
        
        return counters

    def update_likelihood(self, dist, dist_new, labda=0.8):
        assert all(i==j for (i,j) in zip(dist.keys(), dist_new.keys()))
        return {
           key: {i:(j*labda + l*(1-labda)) for ((i,j), (k,l)) in zip(dist[key].items(), dist_new[key].items()) if i == k} 
           for key in dist
        }
    
    def infer_rule_str(self, counters, acc) -> Tuple[List, List]:
        # dd, dl, ld, ll are counts for state transitions,
        # e.g. dd -> dead to dead, etc.
        # TODO: Add threasholds respectively. 
        dd, dl = sum(counters[0].values()), sum(counters[1].values())
        ld, ll = sum(counters[2].values()), sum(counters[3].values())
        
        th_ratio = 0.6
        self.d_th = int(th_ratio * (1-acc/100) * (dd+dl))
        self.l_th = int(th_ratio * (1-acc/100) * (ld+ll))
        
        print(dd, dl, ld, ll, self.d_th, self.l_th, acc)
        
        d_all = counters[0] + counters[1]
        l_all = counters[2] + counters[3]
        
        priors = {
            "d": add_zero_val_key({k:round((counters[0].get(k, 0))/(v), 5) for (k,v) in d_all.items()}),
            "l": add_zero_val_key({k:round((counters[2].get(k, 0))/(v), 5) for (k,v) in l_all.items()}),
        }
        
        likelihood = {
            "d": add_zero_val_key({k:round((v)/(dd+dl), 5) for (k,v) in d_all.items()}),
            "l": add_zero_val_key({k:round((v)/(ld+ll), 5) for (k,v) in l_all.items()}),
        }
        
        posterior = {
           key: add_zero_val_key({i:j*(l**0.1) for ((i,j), (k,l)) in zip(priors[key].items(), likelihood[key].items()) if i == k}) 
           for key in priors.keys()
        }
        
        if self.last_likelihood is None:
            self.last_likelihood = posterior
        else:
            posterior = self.update_likelihood(self.last_likelihood, posterior)
            self.last_likelihood = posterior
        
        from pprint import pprint
        
        # pprint(priors)
        pprint(f"{priors=}\n{likelihood=}\n{posterior=}\n")

        filtered_b = sorted(list(filter(lambda x:x[1]>self.d_th, d_all.items())), key=lambda x:x[0])
        filtered_s = sorted(list(filter(lambda x:x[1]>self.l_th, l_all.items())), key=lambda x:x[0])
        
        self.born = []
        self.survive = []

        list_str = lambda x:list(map(lambda k:str(int(k)), x))
        
        for i,_ in filtered_b:
            if counters[1][i] > 10 * counters[0][i]:
                self.born.append(i)

        for i,_ in filtered_s:
            if counters[3][i] > 10 * counters[2][i]:
                self.survive.append(i)
        
        l_th = 0.7
        
        filtered_l = {
            key: [i[0] for i in sorted(list(filter(lambda x:x[1]>l_th, posterior[key].items())), key=lambda x:x[1], reverse=True)]
            for key in posterior
        }
        
        self.born_l = powerset(set(filtered_l['d']) - set(self.born))
        self.survive_l = powerset(set(filtered_l['l']) - set(self.survive))
        
        ls_born = [sorted(self.born + list(k)) for k in self.born_l]
        ls_survive = [sorted(self.survive + list(k)) for k in self.survive_l]
        
        res = [f"B{''.join(list_str(i))}/S{''.join(list_str(j))}" for (i,j) in product(ls_born, ls_survive)]
        
        return res

    def infer_rule_str2(self, counters, acc) -> Tuple[List, List]:
        # dd, dl, ld, ll are counts for state transitions,
        # e.g. dd -> dead to dead, etc.
        # TODO: Add threasholds respectively. 
        dd, dl = sum(counters[0].values()), sum(counters[1].values())
        ld, ll = sum(counters[2].values()), sum(counters[3].values())
        
        th_ratio = 0.6
        self.dd_th = int(th_ratio * (1-acc/100) * (dd))
        self.dl_th = int(th_ratio * (1-acc/100) * (dl))
        self.ld_th = int(th_ratio * (1-acc/100) * (ld))
        self.ll_th = int(th_ratio * (1-acc/100) * (ll))

        # print(dd, dl, ld, ll, self.dd_th, self.dl_th, self.ld_th, self.ll_th, acc)
        
        dd_all = counters[0]
        dl_all = counters[1]
        ld_all = counters[2]
        ll_all = counters[3]

        filtered_dd = sorted(list(filter(lambda x:x[1]>self.dd_th, dd_all.items())), key=lambda x:x[0])
        filtered_dl = sorted(list(filter(lambda x:x[1]>self.dl_th, dl_all.items())), key=lambda x:x[0])
        filtered_ld = sorted(list(filter(lambda x:x[1]>self.ld_th, ld_all.items())), key=lambda x:x[0])
        filtered_ll = sorted(list(filter(lambda x:x[1]>self.ll_th, ll_all.items())), key=lambda x:x[0])

        self.dd_ls = []
        self.dl_ls = []
        self.ld_ls = []
        self.ll_ls = []
        
        list_str = lambda x:list(map(lambda k:str(int(k)), x))
        
        for i,_ in filtered_dd:
            self.dd_ls.append(i)
        for i,_ in filtered_dl:
            self.dl_ls.append(i)
        for i,_ in filtered_ld:
            self.ld_ls.append(i)
        for i,_ in filtered_ll:
            self.ll_ls.append(i)
        
        return list_str(self.born), list_str(self.survive)

    def test_rule_str(self, rule_str: str, k=10):
        total_loss = 0
        for f in random.choices(self.test_loader.dataset.file_list, k=k):
            ref_traj = np.load(f)
            
            ref_len = ref_traj.shape[0]-1
            
            board = sgl.Board((200, 200), p_pos=0.5)
            board.state = ref_traj[0]
            rule = partial(life_rule, rulestring=rule_str)
        
            sim = sgl.Simulator(board)
            loguru.logger.remove()  # 移除默认配置
            sim.run(rule, ref_len)
            
            pred_ = sim.get_history(exclude_init=True)
            actual_len = min(ref_len, pred_.shape[0])
            
            # TODO: size mismatch bug, triggered when ref_len is small. 
            loss = np.mean((pred_[:actual_len].astype(np.float32) - ref_traj[1:actual_len+1].astype(np.float32))**2).item()
            
            total_loss += loss
        
        return total_loss

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

        # for i in product(*ls):
        for i in range(9):
            ls = ([0 for _ in range(i)] + [1 for _ in range(8-i)])
            random.shuffle(ls)
            
            t0 = tensor(ls[:4] + [0] + ls[4:], dtype=float).reshape(1, 3, 3)
            t1 = tensor(ls[:4] + [1] + ls[4:], dtype=float).reshape(1, 3, 3)
            
            if self.indicator_dataset is None:
                self.indicator_dataset = torch.concatenate([t0, t1])
            else:
                self.indicator_dataset = torch.concatenate([self.indicator_dataset, t0, t1])
        self.indicator_dataset = self.indicator_dataset[:, None, ...].float()
        
        self.indicator_dataset.requires_grad_(False)

    @torch.no_grad()
    def get_rule_from_nn(self, predictor: nn.Module, epsilon: float = 0.1):
        predictor = predictor.to(torch.device("cuda"))
        
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
    
    from model_conv import SimpleCNNSmall_5Layer
    import e2cnn
    
    model = SimpleCNNSmall_5Layer()
    
    param_file = r"D:\Internship\bimsa\result\predictor_life_simple\2025-11-29_23-27-49_small_4_layer_seq_cnn__200-200-B36_S23\best_simple_life_SimpleCNNSmall_5Layer_0.1.0.pth"
    
    rule_str = "B3678/S34678"
    simulator = RuleSimulatorStats(rule_str)
    
    # simulator.load_model(model, param_file)
    
    # stat_ls = simulator.get_transform_stats2()
    
    # simulator.plot_transform_stats(stat_ls, "./")
    
    counters = [Counter({0.0: 16533543, 1.0: 6306252, 3.0: 723050, 4.0: 188583, 5.0: 36750, 6.0: 4743, 2.0: 3187, 7.0: 359, 8.0: 11}), Counter({2.0: 2284373, 1.0: 7565, 3.0: 2726, 0.0: 10}), Counter({0.0: 646039, 3.0: 200116, 4.0: 56385, 5.0: 11486, 6.0: 1478, 7.0: 136, 1.0: 80, 2.0: 44, 8.0: 4}), Counter({1.0: 677315, 2.0: 472346, 0.0: 3408, 3.0: 11})]
    acc = 95
    
    simulator.infer_rule_str(counters, acc)
    
    
    counters = [Counter({0.0: 16533543, 1.0: 6306252, 3.0: 723050, 4.0: 188583, 5.0: 36750, 6.0: 4743, 2.0: 3187, 7.0: 359, 8.0: 11}), Counter({2.0: 2284373, 1.0: 7565, 3.0: 2726, 0.0: 10}), Counter({0.0: 646039, 3.0: 200116, 4.0: 56385, 5.0: 11486, 6.0: 1478, 7.0: 136, 1.0: 80, 2.0: 44, 8.0: 4}), Counter({1.0: 677315, 2.0: 472346, 0.0: 3408, 3.0: 11})]
    acc = 95
    
    simulator.infer_rule_str(counters, acc)
    
    # print(*list(simulator.rule_d.items()), sep="\n")
    