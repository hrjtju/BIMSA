# Python code to implement Conway's Game Of Life
import argparse
import os
import random
import re
from typing import Callable, List, Tuple
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import animation
from loguru import logger
from scipy.signal import convolve2d
from tqdm import tqdm

plt.rcParams["animation.html"] = "jshtml"

import seagull as sgl
import seagull.lifeforms as slf
from seagull.rules import conway_classic, life_rule

from larger_than_life import LTL_FILENAME_PATTERN, generate_neighborhood_matrix



## ==================== Monkey Patch ======================

def life_rule_monkey_patch(X: np.ndarray, rulestring: str) -> np.ndarray:
    """A generalized life rule that accepts a rulestring in B/S notation

    Rulestrings are commonly expressed in the B/S notation where B (birth) is a
    list of all numbers of live neighbors that cause a dead cell to come alive,
    and S (survival) is a list of all the numbers of live neighbors that cause
    a live cell to remain alive.

    Parameters
    ----------
    X : np.ndarray
        The input board matrix
    rulestring : str
        The rulestring in B/S notation

    Returns
    -------
    np.ndarray
        Updated board after applying the rule
    """
    data_dict = _parse_rulestring_monkey_patch(rulestring)
    
    match data_dict["type"]:
        case "classical":
            kernel = generate_neighborhood_matrix(
                neighbor_type="NN" if data_dict["von_neumann"] else "NM",
                radius=1,
                center=False
            )
            neighbors = _count_neighbors_monkey_patch(X, kernel)
        case "LtL":
            kernel = generate_neighborhood_matrix(
                neighbor_type=data_dict["neighborhood"],
                radius=int(data_dict["radius"]),
                center=bool(int(data_dict["middle"]))
            )
            neighbors = _count_neighbors_monkey_patch(X, kernel)
        case _:
            raise NotImplementedError
    
    birth_rule = (X == 0) & (np.isin(neighbors, data_dict["birth_neighbors"]))
    survival_rule = (X == 1) & (np.isin(neighbors, data_dict["survival_neighbors"]))
    
    return birth_rule | survival_rule

def _parse_rulestring_monkey_patch(r: str) -> Tuple[List[int], List[int]]:
    """Parse a rulestring"""
    base_pattern = re.compile("B([0-8]+)?/S([0-8]+)?V?")
    von_neumann = False
    
    if base_pattern.match(r):
        if r.endswith("V"):
            r = r[:-1]
            von_neumann = True
            
        birth, survival = r.split("/")
        birth_neighbors = [int(s) for s in birth if s.isdigit()]
        survival_neighbors = [int(s) for s in survival if s.isdigit()]
        
        data_dict = {
                        "type": "classical", 
                        "birth_neighbors": birth_neighbors,
                        "survival_neighbors": survival_neighbors,
                        "von_neumann": von_neumann
                    }
    elif LTL_FILENAME_PATTERN.match(r):
        data_dict = LTL_FILENAME_PATTERN.match(r).groupdict()
        data_dict["birth_neighbors"] = range(int(data_dict["bmin"]), int(data_dict["bmax"])+1)
        data_dict["survival_neighbors"] = range(int(data_dict["smin"]), int(data_dict["smax"])+1)
        data_dict["type"] = "LtL"
    else:
        msg = f"Rulestring ({r}) must satisfy the pattern {base_pattern}"
        logger.error(msg)
        raise ValueError(msg)

    return data_dict

def _count_neighbors_monkey_patch(X: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Get the number of neighbors in a binary 2-dimensional matrix"""
    n = convolve2d(X, kernel, mode="same", boundary="wrap")
    return n

def board_init_monkey_patch(self, size=(100, 100), p_pos=0.1):
    """Initialize the class

    Parameters
    ----------
    size : array_like of size 2
        Size of the board (default is `(100, 100)`)

    """
    self.size = size
    w, h = self.size
    self.state = np.random.choice([True, False], w*h, p=[p_pos, 1-p_pos]).reshape(w, h)

def simulator_run_monkey_patch(self, rule: Callable, iters: int, **kwargs) -> dict:
        """Run the simulation for a given number of iterations

        Parameters
        ----------
        rule : callable
            Callable that takes in an array and returns an array of the same
            shape.
        iters : int
            Number of iterations to run the simulation.

        Returns
        -------
        dict
           Computed statistics for the simulation run
        """
        layout = self.board.state.copy()

        # Append the initial state
        self.history.append(layout)

        # Run simulation
        for i in range(iters):
            layout = rule(layout, **kwargs)
            
            if np.sum(np.abs(layout ^ self.history[-1])) < 1e-2:
                break
            
            self.history.append(layout)

        self.stats = self.compute_statistics(self.get_history())
        return self.stats


life_rule = life_rule_monkey_patch
sgl.Board.__init__ = board_init_monkey_patch
sgl.Simulator.run = simulator_run_monkey_patch

slf.Pulsar.size = (17, 17)
slf.FigureEight.size = (6, 6)
slf.Glider.size = (3, 3)

## ==================== Code ==============================
class Args:
    size: int
    iters: int
    rule: str

# main() function
def main():

    # Command line args are in sys.argv[1], sys.argv[2] ..
    # sys.argv[0] is the script name itself and can be ignored
    # parse arguments
    parser = argparse.ArgumentParser(description="Runs Conway's Game of Life simulation.")

    # add arguments
    parser.add_argument('--size', type=int, dest="size", help='Width and Height of the grid', required=False, default=200)
    parser.add_argument('--iter', type=int, dest="iters", help='number of iterations', required=False, default=200)
    parser.add_argument('--rule', type=str, dest="rule", help='life rule', required=False, default="B3/S23")
    
    args: Args = parser.parse_args()
    
    bimsa_life_dir = os.environ.get('BIMSA_LIFE_DIR', "./predictor_life_simple/datasets")
    dest_dir = f"{bimsa_life_dir}/{args.size}-{args.iters}-{args.rule.replace('/', '_')}/"
        
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    print(f"Args:\n{args}\n{'='*20}\nDestination: {dest_dir}")
    
    min_max = lambda c,i: (0, args.size - c.size[i])
    rand_loc = lambda c: (random.randint(*min_max(c, 0)), random.randint(*min_max(c, 1)))
    add_rand_loc = lambda c: board.add(c(), loc=rand_loc(c))
    
    for i in tqdm(range(1, 101)):
        board = sgl.Board((args.size, args.size), p_pos=0.6)
        
        for _ in range(int(args.size**0.25)+1):
            add_rand_loc(slf.Pulsar)
        for _ in range(int(args.size**0.5)+1):
            add_rand_loc(slf.Glider)
        for _ in range(int(args.size**0.3)+1):
            add_rand_loc(slf.FigureEight)
        
        rule = partial(life_rule, rulestring=args.rule)
        
        sim = sgl.Simulator(board)
        sim.run(rule, args.iters)
        
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        res = sim.get_history(exclude_init=True)
        # print(res.shape)
        
        np.save(f"{dest_dir}/{i}.npy", res)
    
    # plt.imshow(sim.get_history()[-1])
    # sim.animate()
    # plt.show()

# call main
if __name__ == '__main__':
    main()