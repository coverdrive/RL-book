from __future__ import annotations

import pickle
from typing import Callable, List, Sequence, Tuple, Union
import numpy as np
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

    def put_option_exercise_boundary(
        self, num_steps: int
    ) -> Tuple[Sequence[float], Sequence[float]]:
        
        x: List[float] = []
        y: List[float] = []   
        strike: float = self.strike; expiry: float = self.expiry
        prices: np.ndarray = np.arange(0., strike + 0.1, 0.1)
            
        for step in range(num_steps):
            t: float = step * expiry / num_steps
            cp: Sequence[float] = self.continuation_curve(time_to_expiry=expiry-t,prices=prices)
            ep: Sequence[float] = self.exercise_curve(prices=prices)
            ll: Sequence[float] = [p for p, c, e in zip(prices, cp, ep) if e > c]
            if len(ll) > 0:
                x.append(t)
                y.append(max(ll))
        final: Sequence[Tuple[float, float]] = \
            [(p, self.payoff_func(p)) for p in prices]
        x.append(expiry)
        y.append(max(p for p, e in final if e > 0))
        return x, y

    def save_model(self, filepath) -> None:
        with open(filepath, "wb") as fp:
            pickle.dump(self, fp)

    @classmethod
    def load_model(cls, filepath) -> AlgoWrapper:
        with open(filepath, "rb") as fp:
            obj = pickle.load(fp)
        return obj