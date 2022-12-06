from __future__ import annotations

import argparse
import json
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, Union

import numpy as np

SpotPriceType = Tuple[int, float]

BASEPATH_NUM_STEPS = 100000


@dataclass
class PathFactoryParams:
    expiry: float
    rate: float
    vol: float
    spot_price: float
    num_steps: int
    spot_price_frac: float
    seed: int
    num_paths: int

    @property
    def dt(self) -> float:
        return self.expiry / self.num_steps

    @property
    def vol2(self) -> float:
        return self.vol * self.vol

    def to_file(self, filepath) -> None:
        with open(filepath, "w") as fp:
            json.dump(self.__dict__, fp, allow_nan=False, indent=4)

    @classmethod
    def from_file(cls, filepath) -> PathFactoryParams:
        with open(filepath, "r") as fp:
            params = json.load(fp)
        return cls(**params)


class SimulationPath:
    def __init__(self, params: PathFactoryParams, path: List[SpotPriceType]) -> None:
        self._base_path: List[SpotPriceType] = path
        self.expiry: float = params.expiry
        self.rate: float = params.rate
        self.vol: float = params.vol
        self.base_spot_price: float = params.spot_price
        self.act_spot_price: float = path[0][1]
        self.num_steps: int = params.num_steps
        assert (
            len(self._base_path) == self.num_steps + 1
        ), "Path should have same number of steps as the param that generated it"

    def get_timed_path(self) -> List[Tuple[float, float]]:
        timed_path = []
        for num_steps, price in self._base_path:
            time = num_steps * self.expiry / self.num_steps
            timed_path.append((time, price))
        return timed_path

    def get_timed_triplet_path(self) -> List[Tuple[float,float,float]]:
        timed_triplet_path = []
        timed_path = self.get_timed_path()
        for idx in range(len(timed_path)-1):
            num_steps,price = timed_path[idx]
            time = num_steps * self.expiry / self.num_steps
            _, next_price = timed_path[idx+1]
            timed_triplet_path.append((time, price, next_price))
        return timed_triplet_path

    @classmethod
    def from_filepath(
        cls,
        filepath: str,
        folder_params: PathFactoryParams,
        new_params: PathFactoryParams,
    ):
        path = PriceSimulator.read_path(filepath, folder_params, new_params)
        return cls(new_params, path)


class PriceSimulator:
    def __init__(self, params: PathFactoryParams) -> None:
        self.params = params
        np.random.seed(self.params.seed)

        mean2: float = self.params.spot_price * self.params.spot_price
        var: float = mean2 * self.params.spot_price_frac * self.params.spot_price_frac
        self.log_mean: float = np.log(mean2 / np.sqrt(var + mean2))
        self.log_stdev: float = np.sqrt(np.log(var / mean2 + 1))

    def generate_path(self) -> Sequence[SpotPriceType]:
        price: float = np.random.lognormal(self.log_mean, self.log_stdev)
        yield (0, price)
        for step in range(1, self.params.num_steps + 1):
            m: float = (
                np.log(price)
                + (self.params.rate - self.params.vol2 / 2) * self.params.dt
            )
            v: float = self.params.vol2 * self.params.dt
            price: float = np.exp(np.random.normal(m, np.sqrt(v)))
            yield ((step, price))

    @classmethod
    def read_path(
        self,
        filepath: str,
        folder_params: PathFactoryParams,
        new_params: PathFactoryParams,
    ) -> Sequence[SpotPriceType]:
        path = []
        assert (
            folder_params.num_steps % new_params.num_steps == 0
        ), "We can only simulate integer multiple of original dt"
        every_x_steps = folder_params.num_steps // new_params.num_steps
        with open(filepath, "r") as fp:
            lines = fp.readlines()
        for line in lines[:1] + lines[every_x_steps::every_x_steps]:
            step, price = line.strip().split(",")
            path.append(((int(int(step) / every_x_steps), float(price))))
        return path

    def write_path(self, filepath) -> None:
        with open(filepath, "w") as fp:
            for step, price in self.generate_path():
                fp.write(f"{step}, {price}\n")


class PathFactory:
    def __init__(
        self, params: PathFactoryParams, paths: Sequence[SpotPriceType] = []
    ) -> None:
        self.params: PathFactoryParams = params
        self.paths: List[Sequence[SpotPriceType]] = paths

    @classmethod
    def _filepath(cls, folder_path: str, path_num: int) -> str:
        return f"{folder_path}/path_{path_num}.txt"

    @classmethod
    def _config_path(cls, folder_path: str) -> str:
        return f"{folder_path}/params.json"

    def to_folder(self, folder_path: str, overwrite=False) -> None:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        elif overwrite:
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
        else:
            raise ValueError(f"{folder_path} Already Exists")

        price_sim = PriceSimulator(self.params)
        config_path = self._config_path(folder_path)
        self.params.to_file(config_path)
        print(f"Config: Written to {config_path}")
        for path_num in range(self.params.num_paths):
            filepath: str = self._filepath(folder_path, path_num)
            price_sim.write_path(filepath)
            print(f"Simulation Path {path_num}: Written to {filepath}")

        print(f"{self.params.num_paths} paths written to {folder_path}")

    @classmethod
    def from_folder(cls, folder_path: str, num_steps=None) -> PathFactory:
        folder_params = PathFactoryParams.from_file(cls._config_path(folder_path))
        new_params = deepcopy(folder_params)
        new_params.num_steps = (
            num_steps if num_steps is not None else folder_params.num_steps
        )
        all_paths: List[Sequence[SpotPriceType]] = []
        for path_num in range(folder_params.num_paths):
            filepath: str = cls._filepath(folder_path, path_num)
            path = SimulationPath.from_filepath(filepath, folder_params, new_params)
            all_paths.append(path)
        path_factory = PathFactory(new_params)
        path_factory.paths = all_paths
        return path_factory


def parse_args():
    parser = argparse.ArgumentParser(
        description="Input Params to Price Simulator to Generate Paths"
    )
    parser.add_argument("--expiry", type=float, default=1.0, help="Time to Expiry")
    parser.add_argument("--rate", type=float, default=0.05, help="Risk Free Rate")
    parser.add_argument("--vol", type=float, default=0.25, help="Volatility")
    parser.add_argument(
        "--spot_price", type=float, default=100.0, help="Spot Price at T=0"
    )
    parser.add_argument(
        "--num_paths", type=int, default=100, help="Number of Simulation Paths to Run"
    )
    parser.add_argument(
        "--spot_price_frac",
        type=float,
        default=0.3,
        help="Variation of Spot Price around args.spot at T=0",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random Seed")
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Folder to Store the Generated Runs",
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        help="Overwrite Folder if already exists?",
    )
    args = parser.parse_args()
    return args


def main(args):
    params = PathFactoryParams(
        expiry=args.expiry,
        rate=args.rate,
        vol=args.vol,
        spot_price=args.spot_price,
        num_steps=BASEPATH_NUM_STEPS,
        spot_price_frac=args.spot_price_frac,
        seed=args.seed,
        num_paths=args.num_paths,
    )
    PathFactory(params).to_folder(args.folder_path, overwrite=args.overwrite)


if __name__ == "__main__":
    args = parse_args()
    main(args)
