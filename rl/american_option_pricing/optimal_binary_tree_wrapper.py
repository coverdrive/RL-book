import sys
from typing import Callable, List, Sequence, Tuple
sys.path.append("../../")

from american_option_pricing import AlgoWrapper
from price_simulator import SimulationPath
from rl.chapter8.optimal_exercise_bin_tree import OptimalExerciseBinTree

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
        self.opt_ex_bin_tree: OptimalExerciseBinTree = OptimalExerciseBinTree(
            spot_price=spot_price,
            payoff=payoff_func,
            expiry=expiry,
            rate=rate,
            vol=vol,
            num_steps=self.num_steps
        )

    def train(self, simulation_paths: List[SimulationPath] = None) -> None:
        vf_seq, policy_seq = zip(self.opt_ex_bin_tree.get_opt_vf_and_policy())
        ex_boundary: Sequence[Tuple[float, float]] = self.opt_ex_bin_tree.option_exercise_boundary(policy_seq, is_call=False)

    def predict(self, time_to_expiry: float, price: float) -> float: