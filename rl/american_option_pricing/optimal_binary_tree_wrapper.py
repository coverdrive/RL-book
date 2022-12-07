import numpy as np
import sys
from typing import Callable, Tuple, Iterator, List
sys.path.append("../", "../../")

from american_option_pricing import AlgoWrapper
from price_simulator import SimulationPath
from dynamic_programming import V
from markov_decision_process import Terminal, NonTerminal
from policy import FiniteDeterministicPolicy
from distribution import Constant, Categorical
from finite_horizon import optimal_vf_and_policy

class OptimalBinaryTreeWrapper(AlgoWrapper):
    def __init__(
        self,
        spot_price: float,
        expiry: float,
        rate: float,
        vol: float,
        strike: float,
        payoff_func: Callable[[float], float],
    ) -> None:
        AlgoWrapper.__init__(
            self,
            spot_price = spot_price,
            expiry = expiry,
            rate = rate,
            vol = vol,
            strike = strike,
            payoff_func = payoff_func
        )
        self.num_steps = 1000
        self.vf_seq_best = None
        self.policy_seq_best = None

    def dt(self) -> float:
        """
        Get the time interval size used for discretization 
        """
        return self.expiry / self.num_steps

    def state_price(self, i: int, j: int) -> float:
        """
        Get the price associated with state j of time step i.
        """
        return self.spot_price * np.exp((2 * j - i) * self.vol *
                                        np.sqrt(self.dt()))

    def get_opt_vf_and_policy(self) -> \
            Iterator[Tuple[V[int], FiniteDeterministicPolicy[int, bool]]]:
        """
        Compute the optimal value functions and optimal policies for each time step by backward induction.
        """
        dt: float = self.dt()
        up_factor: float = np.exp(self.vol * np.sqrt(dt))
        up_prob: float = (np.exp(self.rate * dt) * up_factor - 1) / \
            (up_factor * up_factor - 1)
        # this step calls this function from the finite_horizon module.
        return optimal_vf_and_policy( 
            steps=[
                {NonTerminal(j): {
                    True: Constant(
                        (
                            Terminal(-1),
                            self.payoff(self.state_price(i, j))
                        )
                    ),
                    False: Categorical(
                        {
                            (NonTerminal(j + 1), 0.): up_prob,
                            (NonTerminal(j),     0.): 1 - up_prob
                        }
                    )
                } for j in range(i + 1)}
                for i in range(self.num_steps + 1)
            ],
            gamma=np.exp(-self.rate * dt)
        )

    def train(self, simulation_paths: List[SimulationPath] = None) -> None:
        self.vf_seq_best, self.policy_seq_best = zip(*self.get_opt_vf_and_policy())

    def predict(self, time_to_expiry: float, price: float) -> float:
        raise NotImplementedError