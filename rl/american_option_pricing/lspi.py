import sys
import time
from typing import Callable, List, Sequence, Tuple, Union

import numpy as np
from numpy.polynomial.laguerre import lagval

from algo_wrapper import AlgoWrapper
from price_simulator import SimulationPath

import sys
sys.path.append("C://Users//jtuli//Desktop//TJHA//@Stanford//CS 229//FINAL PROJECT CODE//lspi_new_interface//RL-book")
#from rl.function_approx import FunctionApprox, LinearFunctionApprox, Weights
#from function_approx import FunctionApprox, LinearFunctionApprox, Weights

from rl.function_approx import FunctionApprox, LinearFunctionApprox, Weights

TrainingDataType = Tuple[int, float, float]


class LSPI(AlgoWrapper):
    def __init__(
            self,
            spot_price: float,
            expiry: float,
            rate: float,
            vol: float,
            strike: float,
            num_steps: int,
            payoff_func: Callable[ [ float ], float ]) -> None:
        AlgoWrapper.__init__(self, spot_price, expiry, rate, vol, strike, num_steps, payoff_func)
        self.training_iters = 20
        self.trained_model = None


    def train(self, simulation_paths: List[ SimulationPath ]) -> None:
        num_laguerre: int = 4
        epsilon: float = 1e-3

        ident: np.ndarray = np.eye(num_laguerre)
        features: List[ Callable[ [ Tuple[ float, float ] ], float ] ] = [ lambda _: 1. ]
        features += [ (lambda t_s, i=i: np.exp(-t_s[ 1 ] / (2 * self.strike)) *
                                        lagval(t_s[ 1 ] / self.strike, ident[ i ]))
                      for i in range(num_laguerre) ]
        features += [
            lambda t_s: np.cos(-t_s[ 0 ] * np.pi / (2 * self.expiry)),
            lambda t_s: np.log(self.expiry - t_s[ 0 ]) if t_s[ 0 ] != self.expiry else 0.,
            lambda t_s: (t_s[ 0 ] / self.expiry) ** 2
        ]

        print("LOADING TRAIN DATA")
        since = time.time()
        training_data = [ ]
        for simulation_path in simulation_paths:
            training_data.extend(simulation_path.get_timed_triplet_path())
        loading_time = time.time() - since
        print(f"TIME TAKEN TO LOAD : {loading_time:.3f}")
        since = time.time()

        dt: float = self.expiry / self.num_steps
        gamma: float = np.exp(-self.rate * dt)
        num_features: int = len(features)
        states: Sequence[ Tuple[ float, float ] ] = [ (i * dt, s) for
                                                      i, s, _ in training_data ]
        next_states: Sequence[ Tuple[ float, float ] ] = \
            [ ((i + 1) * dt, s1) for i, _, s1 in training_data ]
        feature_vals: np.ndarray = np.array([ [ f(x) for f in features ]
                                              for x in states ])
        next_feature_vals: np.ndarray = np.array([ [ f(x) for f in features ]
                                                   for x in next_states ])
        non_terminal: np.ndarray = np.array(
            [ i < self.num_steps - 1 for i, _, _ in training_data ]
        )
        exer: np.ndarray = np.array([ max(self.strike - s1, 0)
                                      for _, s1 in next_states ])
        wts: np.ndarray = np.zeros(num_features)

        print("STARTING TRAINING")
        print(f"LENGTH OF TRAINING DATA :{len(training_data)}")

        for _ in range(self.training_iters):
            a_inv: np.ndarray = np.eye(num_features) / epsilon
            b_vec: np.ndarray = np.zeros(num_features)
            cont: np.ndarray = np.dot(next_feature_vals, wts)
            cont_cond: np.ndarray = non_terminal * (cont > exer)
            for i in range(len(training_data)):
                phi1: np.ndarray = feature_vals[ i ]
                phi2: np.ndarray = phi1 - \
                                   cont_cond[ i ] * gamma * next_feature_vals[ i ]
                temp: np.ndarray = a_inv.T.dot(phi2)
                a_inv -= np.outer(a_inv.dot(phi1), temp) / (1 + phi1.dot(temp))
                b_vec += phi1 * (1 - cont_cond[ i ]) * exer[ i ] * gamma
            wts = a_inv.dot(b_vec)

        self.trained_model = LinearFunctionApprox.create(
            feature_functions=features,
            weights=Weights.create(wts)
        )

        training_time = time.time() - since
        print(f"TIME TAKEN TO TRAIN : {training_time:.3f}")

    def predict(self, time_to_expiry: float, price: float) -> float:
        return self.trained_model.evaluate([ (self.expiry - time_to_expiry, price) ])[ 0 ]

    def save_model(self, save_path: str):
        pass

    def load_model(self, load_path: str):
        pass
