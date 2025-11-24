import argparse
import datetime
import os
import random

import numpy as np
import toml
import torch
from torch import nn as nn

from dataloader import LifeGameDataset
from rule_simulator import RuleSimulator
from train_test import train_and_evaluate
import model_conv
from model_conv import GroupEquivariantCNN

if __name__ == "__main__":
    
    # reads the command line arguments
    in_profile = argparse.ArgumentParser(description="Train the Predictor Life model")
    
    in_profile.add_argument("-p", "--hyperparameters", type=str, default="./predictor_life_simple/hyperparams/baseline.toml", help="Path to hyperparameters file")
    in_profile.add_argument("-r", "--sysRule", dest="data_rule", type=str, default="B3/S23", help="Life rules")
    in_profile.add_argument("-i", "--dataIter", dest="data_iters", type=int, default=200, help="Iterations within each data file")
    in_profile.add_argument("-w", "--sysSize", dest="sys_size", type=int, default=200, help="System size")
    
    in_profile_args = in_profile.parse_args()

    print(in_profile_args.hyperparameters, end='\n\n')
    args_dict = toml.load(in_profile_args.hyperparameters) | dict(in_profile_args._get_kwargs())
    print(args_dict)
    print("Starting training...")
    
    args_dict["wandb"]["turn_on"] = False
    
    bimsa_life_dir = os.environ.get('BIMSA_LIFE_DIR', "./predictor_life_simple/datasets")
    rule_data_str = f"{args_dict['sys_size']}-{args_dict['data_iters']}-{args_dict['data_rule'].replace('/', '_')}"
    dataset_dir = f"{bimsa_life_dir}/{rule_data_str}/"
    
    start_time_str = f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}".split('.')[0]
        
    # initialize dataset and dataloader with only one visible trajectory. 
    train_dataset = LifeGameDataset(dataset_dir, split="train")
    test_dataset = LifeGameDataset(dataset_dir, split="test")
    
    dataloader_args = {
        "batch_size": args_dict["dataloader"]["train_batch_size"],
        "shuffle": args_dict["dataloader"]["train_shuffle"],
        "num_workers": args_dict["dataloader"]["train_num_workers"],
    }
    
    # initialize RuleSimulator
    simulator = RuleSimulator(n=3)
    
    # initialize predictor model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_class = getattr(model_conv, args_dict["model"]["name"])
    
    # Initialize model & Move model to device
    model: nn.Module|GroupEquivariantCNN = model_class().to(device)
    
    for round_id in range(1, 21):
        print(f"\n{'='*30}\n{' '*15}ROUND {round_id}\n{'='*30}\n")
        
        # while prediction loss and simulation loss is not converged:
        train_loader = train_dataset.get_dataloader(**dataloader_args)
        test_loader = test_dataset.get_dataloader(**dataloader_args)
        
        # train predictor on visible trajectories.
        model, train_loss, evaluate_acc = train_and_evaluate(device, 
                                                             model, 
                                                             train_loader, 
                                                             test_loader, 
                                                             round_id, 
                                                             rule_data_str,
                                                             dataset_dir,
                                                             start_time_str,
                                                             args_dict)
        
        #! if first time training: deterrmine equivariant condtraints.
        # update rules in RuleSimulator from predictor.
        simulator.get_rule_from_nn(model)
        
        # randomly select one of the invisible trajectories to simulate.
        traj_gound_truth = np.load(random.choice(list(set(train_dataset.visible_trajectories))))
        
        # evaluate predictor on one of the visible trajectories.
        traj_simulated = simulator.predict_seq(torch.tensor(traj_gound_truth[0:1]), t=traj_gound_truth.shape[0]-1)
        
        # calculate acc.
        sim_acc = (np.abs(traj_simulated.numpy() - np.array(traj_gound_truth, dtype=np.float32)).mean())
        
        print(f"Final test score: {sim_acc}")
        
        # if total loss is converged AND accuracy is over 99% : break.
        if sim_acc > 0.99 and train_loss < 0.01 and evaluate_acc > 0.95:
            print(f"\nTERMINATED with SimulationAcc: {sim_acc:.4f}\n")
            break
        else:
            # randomly add one trajectory 
            train_dataset.add_trajectory()
        
    
    # save predictor model.
    torch.save(model.state_dict(), "predictor_model.pth")
    
    # save rules.
    with open("extracted_rules.txt", "w") as f:
        for k, v in simulator.rule_d.items():
            f.write(f"Input:\n{k}\nOutput:\n{v}\n\n")