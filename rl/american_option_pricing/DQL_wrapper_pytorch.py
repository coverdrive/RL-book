from __future__ import annotations
from typing import Callable, Sequence, Tuple, List
import numpy as np
import random
from scipy.stats import norm
import sys
sys.path.extend(["../", "../../"])
from rl.function_approx import DNNApprox, LinearFunctionApprox, \
    FunctionApprox, DNNSpec, AdamGradient, Weights
from random import randrange
from numpy.polynomial.laguerre import lagval
from rl.chapter8.optimal_exercise_bin_tree import OptimalExerciseBinTree
from rl.markov_process import NonTerminal
from rl.gen_utils.plot_funcs import plot_list_of_curves

import pickle
import time
from typing import Callable, List, Sequence, Tuple, Union
import matplotlib.pyplot as plt
from price_simulator import SimulationPath, PathFactory
from algo_wrapper import AlgoWrapper

import torch
import torch.nn as nn
from tqdm import tqdm


class DQN(nn.Module):

    def __init__(self,layers,strike,expiry):
        super(DQN,self).__init__()
        
        # setup feature functions
        num_laguerre: int = 2
        ident: np.ndarray = np.eye(num_laguerre)
        self.feature_functions = [lambda _: 1.]
        self.feature_functions += [(lambda t_s, i=i: np.exp(-t_s[1] / (2 * strike)) *
                      lagval(t_s[1] / strike, ident[i]))
                     for i in range(num_laguerre)]
        self.feature_functions += [
            lambda t_s: np.cos(-t_s[0] * np.pi / (2 * expiry)),
            lambda t_s: np.log(expiry - t_s[0]) if t_s[0] != expiry else 0.,
            lambda t_s: (t_s[0] / expiry) ** 2
        ]
        # setup DQN architecture
        self.layers = []
        layers = [len(self.feature_functions)]+layers
        self.layers.extend([nn.Linear(layers[0],layers[1])])
        
        for prev_layer,layer in zip(layers[1:-1],layers[2:]):
            self.layers.extend([nn.ReLU(),nn.Linear(prev_layer,layer),])
        self.layers.append(nn.Softplus(beta=-1))
        self.layers.append(nn.Linear(layers[-1],1))
        # layers setup as nn.Sequential, final activation layer in forward()
        self.layers = nn.Sequential(*self.layers)
        print(self.layers)
        
    def forward(self,x):
        # extract features from (t,s)
        x = torch.tensor([[f(i) for f in self.feature_functions] for i in x],dtype=torch.float32)
        # pass features through DQN
        x = self.layers(x)
        # pass it through the final activation layer
        # return torch.log(1+torch.exp(-x))
        return x

class FA_obj():

    def __init__(self,dqn,aug_weight):
        self.dqn = dqn
        self.optimizer = torch.optim.Adam(
            params=self.dqn.parameters(),
            lr=1e-1,
            betas=(0.9,0.999),
            weight_decay=1e-2
        )
        self.aug_weight = aug_weight
        self.loss = nn.MSELoss()

    def evaluate(self,x):
        self.dqn.eval()
        with torch.no_grad():
            pred = self.dqn(x)
        return pred.item()

    def update(self,replay_data,aug_data=None,verbose=False):
        
        x = [item[0] for item in replay_data]
        y = torch.tensor([item[1] for item in replay_data],dtype=torch.float32).unsqueeze(dim=1)
        
        self.dqn.train()
        self.optimizer.zero_grad()
        if(aug_data is None):
            pred = self.dqn(x)
            loss = ((y-pred)**2).mean()
        else:
            x_aug = [item[0] for item in aug_data]
            y_aug = torch.tensor([item[1] for item in aug_data],dtype=torch.float32).unsqueeze(dim=1)
            pred = self.dqn(x)
            pred_aug = self.dqn(x_aug)
            loss_vec = torch.cat([(y-pred)**2,self.aug_weight*(y_aug-pred_aug)**2],dim=0)
            loss = loss_vec.mean()
        loss.backward()
        if(verbose):
            print(loss)
        for name, param in self.dqn.named_parameters():
            if(not(torch.isfinite(param.grad).all())):
                print("NF")
                return
        # torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 10)
        # print(loss)
        self.optimizer.step()

class DQL_P(AlgoWrapper):
    
    def __init__(self,
                spot_price: float,
                expiry: float,
                rate: float,
                vol: float,
                strike: float,
                num_steps:float,
                payoff_func: Callable[[float], float],
                aug_weight=1
                ):
        
        # setup standard environment parameters
        AlgoWrapper.__init__(self,spot_price,expiry,rate,vol,strike,num_steps,payoff_func)
        
        # setup DQL
        self.dqn = DQN(layers=[6,1],strike=strike,expiry=expiry)
        self.fa = FA_obj(self.dqn,aug_weight)

    def train(self,
              training_data,
              num_steps,
#               simulation_paths: List[SimulationPath],
              training_iters: int):
        
        dt: float = self.expiry / num_steps
        gamma: float = np.exp(-self.rate * dt)
        
        augmented_data = []; aug_steps = 100*num_steps
        for idx in range(aug_steps-1):
            augmented_data.append(((idx*self.expiry/aug_steps,0),self.strike))
        random.shuffle(augmented_data)
#         print("LOADING TRAIN DATA")
#         since = time.time()
#         training_data = []
#         for simulation_path in simulation_paths:
#             training_data.extend(simulation_path.get_timed_triplet_path())
#         loading_time = time.time()-since
#         print(f"TIME TAKEN TO LOAD : {loading_time:.3f}")
        
        since = time.time()
        replay_buffer = []; replay_buffer_size=10
        aug_buffer = []; aug_buffer_size=10
        print("STARTING TRAINING")
        for idx in tqdm(range(training_iters)):
            verbose=(idx%(training_iters//10)==0)
            
            t, s, s1 = training_data[randrange(len(training_data))]
            x_val: Tuple[float, float] = (t, s)
            val: float = self.payoff_func(s1)
            if t/dt < num_steps-2:
                _temp = self.fa.evaluate([(t + dt, s1)])
                val = max(val, _temp)
            
            # update using replay buffer training data
            y_val: float = gamma * val
            if(len(replay_buffer)<replay_buffer_size):
                replay_buffer.append((x_val,y_val))
            else:
                replay_buffer[idx%(len(replay_buffer))] = (x_val,y_val)

            # update using both replay and augmented buffer training data
            if(np.random.binomial(1,1)):
                aug_data = augmented_data[idx%len(augmented_data)]
                if(len(aug_buffer)<aug_buffer_size):
                    aug_buffer.append(aug_data)
                else:
                    aug_buffer[np.random.choice(len(aug_buffer))] = aug_data
                self.fa.update(replay_buffer,aug_buffer,verbose)
            else:
                self.fa.update(replay_buffer,None,verbose)
        training_time = time.time()-since
        print(f"TIME TAKEN TO TRAIN : {training_time:.3f}")
    
    def predict(self,
                time_to_expiry:float,
                price:float)->float:
        return self.fa.evaluate([(self.expiry-time_to_expiry,price)])