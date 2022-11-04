from __future__ import annotations

import pickle
from typing import Callable, List, Sequence, Tuple, Union

from price_simulator import SimulationPath


class AlgoWrapper:
    def __init__(
        self,
        spot_price: float,
        expiry: float,
        rate: float,
        vol: float,
        strike: float,
        payoff_func: Callable[[float], float],
    ) -> None:
        self.spot_price = spot_price
        self.expiry = expiry
        self.rate = rate
        self.vol = vol
        self.strike = strike
        self.payoff_func = payoff_func

    def train(self, simulation_paths: List[SimulationPath]) -> None:
        raise NotImplementedError

    def predict(self, time_to_expiry: float, price: float) -> float:
        raise NotImplementedError

    def continuation_curve(
        self, time_to_expiry: float, prices: Sequence[float]
    ) -> List[float]:
        return [self.predict(time_to_expiry, p) for p in prices]

    def exercise_curve(self, prices: Sequence[float]) -> Sequence[float]:
        return [self.payoff_func(p) for p in prices]

    def option_price_curve(
        self, time_to_expiry: float, prices: Sequence[float]
    ) -> List[float]:
        continuation_payoff = self.continuation_curve(time_to_expiry, prices)
        exercise_payoff = self.exercise_curve(prices)
        return [max(x, y) for x, y in zip(continuation_payoff, exercise_payoff)]

    def time_evolution_curve(
        self, times_to_expiry: List[float], price: float
    ) -> List[float]:
        exercise_payoff = self.payoff_func(price)
        return [max(exercise_payoff, self.predict(t, price)) for t in times_to_expiry]

    def save_model(self, filepath) -> None:
        with open(filepath, "wb") as fp:
            pickle.dump(self, fp)

    @classmethod
    def load_model(cls, filepath) -> OptimizerAlgoWrapper:
        with open(filepath, "rb") as fp:
            obj = pickle.load(fp)
        return obj