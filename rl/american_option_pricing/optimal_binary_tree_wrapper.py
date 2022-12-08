import numpy as np
import sys
from typing import Callable, Tuple, Iterator, List
sys.path.append("../")
sys.path.append("../../")

from algo_wrapper import AlgoWrapper
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
        prices = np.asarray([self.state_price(step, j) for j in range(step+1)])

        # Compute the absolute difference between the prices and required price
        diff = abs(prices - price)

        # The state we want is the closest to the required price
        return np.argmin(diff)

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
                            self.payoff_func(self.state_price(i, j))
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
        return self.vf_seq_best[step][NonTerminal(state=state)]

def main():
    spot_price_val: float = 100.0 #100.0
    strike_val: float = 100.0 #100.0
    expiry_val: float = 1.0 #1.0
    rate_val: float = 0.05 #0.05
    vol_val: float = 0.25 #0.25
    opt_payoff: Callable[[float],float] = lambda x: max(strike_val - x, 0)

    print("Initialize the OptimalBinaryTreeWrapper")
    optBinTree = OptimalBinaryTreeWrapper(
        spot_price=spot_price_val,
        expiry=expiry_val,
        rate=rate_val,
        vol=vol_val,
        strike=strike_val,
        payoff_func=opt_payoff
    )

    print("Training the optimal binary tree")
    optBinTree.train()

    print("vf_seq_best[0][NonTerminal(state=0)]: ", optBinTree.vf_seq_best[0][NonTerminal(state=0)])

    time_to_expiry = 0.5
    price = 100.0
    prediction = optBinTree.predict(time_to_expiry=time_to_expiry, price=price)
    print("Prediction value with time_to_expiry: ", time_to_expiry, " and price: ", price, " is: ", prediction)

if __name__ == "__main__":
    main()
