# Python code to implement Conway's Game Of Life
import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from numba import njit
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import animation

plt.rcParams["animation.html"] = "jshtml"

import seagull as sgl
import seagull.lifeforms as slf
from seagull.rules import conway_classic, life_rule

from tqdm import tqdm

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
    sim.animate()
    plt.show()

# call main
if __name__ == '__main__':
    main()