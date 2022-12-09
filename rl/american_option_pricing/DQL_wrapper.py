from __future__ import annotations

import random
import sys
from typing import Callable, List, Sequence, Tuple

import numpy as np
from scipy.stats import norm

sys.path.extend(["../", "../../"])
import pickle
import time
from random import randrange
from typing import Callable, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from numpy.polynomial.laguerre import lagval
from rl.chapter8.optimal_exercise_bin_tree import OptimalExerciseBinTree
from rl.function_approx import (AdamGradient, DNNApprox, DNNSpec,
                                FunctionApprox, LinearFunctionApprox, Weights)
from rl.gen_utils.plot_funcs import plot_list_of_curves
from rl.markov_process import NonTerminal

from algo_wrapper import AlgoWrapper
from price_simulator import PathFactory, SimulationPath


class DQL_Learner(AlgoWrapper):
    
    def __init__(self,
                spot_price: float,
                expiry: float,
                rate: float,
                vol: float,
                strike: float,
                num_steps: int,
                payoff_func: Callable[[float], float],
                dnn_params : dict = None,
                fa_spec : dict = None
                ):
        
        # setup standard environment parameters
        AlgoWrapper.__init__(self,spot_price,expiry,rate,vol,strike,num_steps,payoff_func)
        
        # setup DQL
        #------------------------------------------------------------------------
        # setup dnn specs
        if(dnn_params is None):
            self.dnn_params = {}
            self.dnn_params['neurons']:Sequence[int] = [6]
            self.dnn_params['bias']=True
            self.dnn_params['hidden_activation']=lambda x: np.log(1 + np.exp(-x))
            self.dnn_params['hidden_activation_deriv']=lambda y: np.exp(-y) - 1
            self.dnn_params['output_activation']=lambda x: x
            self.dnn_params['output_activation_deriv']=lambda y: np.ones_like(y)
        else:
            self.dnn_params = dnn_params
            
        self.dnn_spec: DNNSpec = DNNSpec(**self.dnn_params)

        # setup func approx specs
        if(fa_spec is None):
            self.fa_spec = {}
            num_laguerre: int = 2
            ident: np.ndarray = np.eye(num_laguerre)
            self.fa_spec['feature_functions']: List[Callable[[Tuple[float, float]], float]] = [lambda _: 1.]
            self.fa_spec['feature_functions'] += [(lambda t_s, i=i: np.exp(-t_s[1] / (2 * strike)) *
                          lagval(t_s[1] / strike, ident[i]))
                         for i in range(num_laguerre)]
            self.fa_spec['feature_functions'] += [
                lambda t_s: np.cos(-t_s[0] * np.pi / (2 * expiry)),
                lambda t_s: np.log(expiry - t_s[0]) if t_s[0] != expiry else 0.,
                lambda t_s: (t_s[0] / expiry) ** 2
            ]
            self.fa_spec['adam_gradient']=AdamGradient(
                learning_rate=0.01,
                decay1=0.9,
                decay2=0.999
            )
            self.fa_spec['regularization_coeff']=1e-2
        else:
            self.fa_spec = fa_spec
    
        self.fa: DNNApprox[Tuple[float, float]] = DNNApprox.create(dnn_spec=self.dnn_spec,**self.fa_spec)
        #--------------------------------------------------------------------------
    
    def train(self,
              simulation_paths: List[SimulationPath],
              training_epochs: int = 10):
        
        dt: float = self.expiry / self.num_steps
        gamma: float = np.exp(-self.rate * dt)
        
        print("LOADING TRAIN DATA")
        since = time.time()
        training_data = []
        for simulation_path in simulation_paths:
            training_data.extend(simulation_path.get_timed_triplet_path())
        loading_time = time.time()-since
        print(f"TIME TAKEN TO LOAD : {loading_time:.3f}")
        since = time.time()

        print("STARTING TRAINING")
        print(f"LENGTH OF TRAINING DATA :{len(training_data)}")
        for idx in range(training_epochs):
            if(idx%(training_epochs/10)==0):
                print(f'{idx}/{training_epochs}')
            for _ in range(len(training_data)):
                t, s, s1 = training_data[randrange(len(training_data))]
                x_val: Tuple[float, float] = (t, s)
                val: float = self.payoff_func(s1)
                if t/dt < self.num_steps - 1:
                    val = max(val, self.fa.evaluate([(t + dt, s1)])[0])
                y_val: float = gamma * val
                self.fa = self.fa.update([(x_val, y_val)])
        
        training_time = time.time()-since
        print(f"TIME TAKEN TO TRAIN : {training_time:.3f}")
    
    def predict(self,
                time_to_expiry:float,
                price:float)->float:
        return self.fa.evaluate([(self.expiry-time_to_expiry,price)])[0]
    
    def save_model(self,
                   save_path: str):
        pass
    
    def load_model(self,
                   load_path: str):
        pass