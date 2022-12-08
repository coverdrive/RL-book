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

    def get_step_from_time_to_expiry(self, time_to_expiry: float) -> int:
        """
        Get the step (i) from the given time to expiry.

        Say that we have the original expiry as 1 year and are given
        time_to_expiry as 4 months, the step value (i) would be 
        1 - 4(months)/12(months) times the number of steps.
        """
        return int(self.num_steps * (1 - time_to_expiry / self.expiry))

    def get_state_from_price(self, price: float, step: int) -> int:
        """
        Given the price and the time step, find the j'th state.
        We compute this by finding the prices at that time step (time step i has i+1 prices).
        We then return the j for which prices[j] is closest to the price.
        """
        # Get the prices available at time step "step".
        prices = [self.state_price(step, j) for j in range(step+1)]

        # Find the price in prices that is closest to the required price
        min_diff = sys.maxint

        # State is the value of j that is closest in price to the 
        # required price
        state = 0
        for j in range(len(prices)):
            p = prices[j]
            if abs(price - p) < min_diff:
                min_diff = price-p
                state = j

        return state

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
        """
        Perform the backwards induction to get the optimal value function for all states (i, j)
        """
        self.vf_seq_best, self.policy_seq_best = zip(*self.get_opt_vf_and_policy())

    def predict(self, time_to_expiry: float, price: float) -> float:
        """
        Predict the continuation value given the time to expiry and the price
        """
        # Get the step (i) from the time to expiry
        step = self.get_step_from_time_to_expiry(time_to_expiry=time_to_expiry)

        # If it is the last step, return the price because there is no continuation value
        if step == self.num_steps - 1:
            return price

        # Get the j corresponding to the price in time step "step"
        state = self.get_state_from_price(price=price, step=step)

        # Since we want the continuation value, we put NonTerminal(True)
        return self.vf_seq_best[step][state][NonTerminal(True)]