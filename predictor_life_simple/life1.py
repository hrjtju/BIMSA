# Python code to implement Conway's Game Of Life
import argparse
import os
import random
import re
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import animation
from loguru import logger
from scipy.signal import convolve2d

plt.rcParams["animation.html"] = "jshtml"

import seagull as sgl
import seagull.lifeforms as slf
from seagull.rules import conway_classic, life_rule

from tqdm import tqdm


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
    birth_req, survival_req, von_neumann = _parse_rulestring_monkey_patch(rulestring)
    neighbors = _count_neighbors_monkey_patch(X, von_neumann)
    birth_rule = (X == 0) & (np.isin(neighbors, birth_req))
    survival_rule = (X == 1) & (np.isin(neighbors, survival_req))
    return birth_rule | survival_rule

def _parse_rulestring_monkey_patch(r: str) -> Tuple[List[int], List[int]]:
    """Parse a rulestring"""
    pattern = re.compile("B([0-8]+)?/S([0-8]+)?V?")
    von_neumann = False
    
    if pattern.match(r):
        if r.endswith("V"):
            r = r[:-1]
            von_neumann = True
            
        birth, survival = r.split("/")
        birth_neighbors = [int(s) for s in birth if s.isdigit()]
        survival_neighbors = [int(s) for s in survival if s.isdigit()]
    else:
        msg = f"Rulestring ({r}) must satisfy the pattern {pattern}"
        logger.error(msg)
        raise ValueError(msg)

    return birth_neighbors, survival_neighbors, von_neumann

def _count_neighbors_monkey_patch(X: np.ndarray, von_neumann: bool) -> np.ndarray:
    """Get the number of neighbors in a binary 2-dimensional matrix"""
    if von_neumann:
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    else:
        kernel = np.ones((3, 3))
        
    n = convolve2d(X, kernel, mode="same", boundary="wrap") - X
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

life_rule = life_rule_monkey_patch
sgl.Board.__init__ = board_init_monkey_patch

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
    
    for i in range(1, 21):
        board = sgl.Board((args.size, args.size), p_pos=(0.2 + i * 0.02))
        
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
        
        np.save(f"{dest_dir}/{i}.npy", sim.get_history(exclude_init=True))
    
    # plt.imshow(sim.get_history()[-1])
    # sim.animate()
    # plt.show()

# call main
if __name__ == '__main__':
    main()