'''An interface for different kinds of function approximations
(tabular, linear, DNN... etc), with several implementations.'''

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace, field
import itertools
import numpy as np
from operator import itemgetter
from scipy.interpolate import splrep, BSpline
from typing import (Callable, Dict, Generic, Iterator, Iterable, List,
                    Mapping, Optional, Sequence, Tuple, TypeVar)
import rl.iterate as iterate

X = TypeVar('X')
SMALL_NUM = 1e-6


class FunctionApprox(ABC, Generic[X]):
    '''Interface for function approximations.
    An object of this class approximates some function X ↦ ℝ in a way
    that can be evaluated at specific points in X and updated with
    additional (X, ℝ) points.
    '''

    @abstractmethod
    def representational_gradient(self, x_value: X) -> FunctionApprox[X]:
        '''Computes the gradient of the self FunctionApprox with respect
        to the parameters in the internal representation of the
        FunctionApprox, i.e., computes Gradient with respect to internal
        parameters of expected value of y for the input x, where the
        expectation is with respect tp the FunctionApprox's model of
        the probability distribution of y|x. The gradient is output
        in the form of a FunctionApprox whose internal parameters are
        equal to the gradient values.
        '''

    @abstractmethod
    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        '''Computes expected value of y for each x in
        x_values_seq (with the probability distribution
        function of y|x estimated as FunctionApprox)
        '''

    def __call__(self, x_value: X) -> float:
        return self.evaluate([x_value]).item()

    @abstractmethod
    def update(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> FunctionApprox[X]:

        '''Update the internal parameters of the FunctionApprox
        based on incremental data provided in the form of (x,y)
        pairs as a xy_vals_seq data structure
        '''

    @abstractmethod
    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> FunctionApprox[X]:
        '''Assuming the entire data set of (x,y) pairs is available
        in the form of the given input xy_vals_seq data structure,
        solve for the internal parameters of the FunctionApprox
        such that the internal parameters are fitted to xy_vals_seq.
        Since this is a best-fit, the internal parameters are fitted
        to within the input error_tolerance (where applicable, since
        some methods involve a direct solve for the fit that don't
        require an error_tolerance)
        '''

    @abstractmethod
    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        '''Is this function approximation within a given tolerance of
        another function approximation of the same type?
        '''

    def argmax(self, xs: Iterable[X]) -> X:
        '''Return the input X that maximizes the function being approximated.
        Arguments:
          xs -- list of inputs to evaluate and maximize, cannot be empty
        Returns the X that maximizes the function this approximates.
        '''
        return list(xs)[np.argmax(self.evaluate(xs))]

    def rmse(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> float:
        '''The Root-Mean-Squared-Error between FunctionApprox's
        predictions (from evaluate) and the associated (supervisory)
        y values
        '''
        x_seq, y_seq = zip(*xy_vals_seq)
        errors: np.ndarray = self.evaluate(x_seq) - np.array(y_seq)
        return np.sqrt(np.mean(errors * errors))

    def iterate_updates(
        self,
        xy_seq_stream: Iterator[Iterable[Tuple[X, float]]]
    ) -> Iterator[FunctionApprox[X]]:
        '''Given a stream (Iterator) of data sets of (x,y) pairs,
        perform a series of incremental updates to the internal
        parameters (using update method), with each internal
        parameter update done for each data set of (x,y) pairs in the
        input stream of xy_seq_stream
        '''
        return itertools.accumulate(
            xy_seq_stream,
            lambda fa, xy: fa.update(xy),
            initial=self
        )

    def representational_gradient_stream(
        self,
        x_values_seq: Iterable[X]
    ) -> Iterator[FunctionApprox[X]]:
        for x_val in x_values_seq:
            yield self.representational_gradient(x_val)


@dataclass(frozen=True)
class Dynamic(FunctionApprox[X]):
    '''A FunctionApprox that works exactly the same as exact dynamic
    programming. Each update for a value in X replaces the previous
    value at X altogether.

    Fields:
    values_map -- mapping from X to its approximated value
    '''

    values_map: Mapping[X, float]

    def representational_gradient(self, x_value: X) -> Dynamic[X]:
        return Dynamic({x_value: 1.0})

    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        '''Evaluate the function approximation by looking up the value in the
        mapping for each state.

        Will raise an error if an X value has not been seen before and
        was not initialized.

        '''
        return np.array([self.values_map[x] for x in x_values_seq])

    def update(self, xy_vals_seq: Iterable[Tuple[X, float]]) -> Dynamic[X]:
        '''Update each X value by replacing its saved Y with a new one. Pairs
        later in the list take precedence over pairs earlier in the
        list.

        '''
        new_map = dict(self.values_map)
        for x, y in xy_vals_seq:
            new_map[x] = y

        return replace(self, values_map=new_map)

    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> Dynamic[X]:
        return replace(self, value_map=dict(xy_vals_seq))

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        '''This approximation is within a tolerance of another if the value
        for each X in both approximations is within the given
        tolerance.

        Raises an error if the other approximation is missing states
        that this approximation has.

        '''
        if not isinstance(other, Dynamic):
            return False

        return all(abs(self.values_map[s] - other.values_map[s]) <= tolerance
                   for s in self.values_map)


@dataclass(frozen=True)
class Tabular(FunctionApprox[X]):
    '''Approximates a function with a discrete domain (`X'), without any
    interpolation. The value for each `X' is maintained as a weighted
    mean of observations by recency (managed by
    `count_to_weight_func').

    In practice, this means you can use this to approximate a function
    with a learning rate α(n) specified by count_to_weight_func.

    If `count_to_weight_func' always returns 1, this behaves the same
    way as `Dynamic'.

    Fields:
    values_map -- mapping from X to its approximated value
    counts_map -- how many times a given X has been updated
    count_to_weight_func -- function for how much to weigh an update
      to X based on the number of times that X has been updated

    '''

    values_map: Mapping[X, float] = field(default_factory=lambda: {})
    counts_map: Mapping[X, int] = field(default_factory=lambda: {})
    count_to_weight_func: Callable[[int], float] = \
        field(default_factory=lambda: lambda n: 1.0 / n)

    def representational_gradient(self, x_value: X) -> Tabular[X]:
        return Tabular({x_value: 1.0})

    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        '''Evaluate the approximation at each given X.

        If an X has not been seen before, will return 0.0.
        '''
        return np.array([self.values_map.get(x, 0.) for x in x_values_seq])

    def update(self, xy_vals_seq: Iterable[Tuple[X, float]]) -> Tabular[X]:
        '''Update the approximation with the given points.

        Each X keeps a count n of how many times it was updated, and
        each subsequent update is discounted by
        count_to_weight_func(n), which defines our learning rate.

        '''
        values_map: Dict[X, float] = dict(self.values_map)
        counts_map: Dict[X, int] = dict(self.counts_map)

        for x, y in xy_vals_seq:
            counts_map[x] = counts_map.get(x, 0) + 1
            weight: float = self.count_to_weight_func(counts_map.get(x, 0))
            values_map[x] = weight * y + (1 - weight) * values_map.get(x, 0.)

        return replace(
            self,
            values_map=values_map,
            counts_map=counts_map
        )

    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> Tabular[X]:
        values_map: Dict[X, float] = {}
        counts_map: Dict[X, int] = {}
        for x, y in xy_vals_seq:
            counts_map[x] = counts_map.get(x, 0) + 1
            weight: float = self.count_to_weight_func(counts_map.get(x, 0))
            values_map[x] = weight * y + (1 - weight) * values_map.get(x, 0.)
        return replace(
            self,
            values_map=values_map,
            counts_map=counts_map
        )

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        if isinstance(other, Tabular):
            return\
                all(abs(self.values_map[s] - other.values_map[s]) <= tolerance
                    for s in self.values_map)

        return False


@dataclass(frozen=True)
class BSplineApprox(FunctionApprox[X]):
    feature_function: Callable[[X], float]
    degree: int
    knots: np.ndarray = field(default_factory=lambda: np.array([]))
    coeffs: np.ndarray = field(default_factory=lambda: np.array([]))

    def get_feature_values(self, x_values_seq: Iterable[X]) -> Sequence[float]:
        return [self.feature_function(x) for x in x_values_seq]

    def representational_gradient(self, x_value: X) -> BSplineApprox[X]:
        feature_val: float = self.feature_function(x_value)
        eps: float = 1e-6
        one_hots: np.array = np.eye(len(self.coeffs))
        return replace(
            self,
            coeffs=np.array([(
                BSpline(
                    self.knots,
                    c + one_hots[i] * eps,
                    self.degree
                )(feature_val) -
                BSpline(
                    self.knots,
                    c - one_hots[i] * eps,
                    self.degree
                )(feature_val)
            ) / (2 * eps) for i, c in enumerate(self.coeffs)]))

    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        spline_func: Callable[[Sequence[float]], np.ndarray] = \
            BSpline(self.knots, self.coeffs, self.degree)
        return spline_func(self.get_feature_values(x_values_seq))

    def update(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> BSplineApprox[X]:
        x_vals, y_vals = zip(*xy_vals_seq)
        feature_vals: Sequence[float] = self.get_feature_values(x_vals)
        sorted_pairs: Sequence[Tuple[float, float]] = \
            sorted(zip(feature_vals, y_vals), key=itemgetter(0))
        new_knots, new_coeffs, _ = splrep(
            [f for f, _ in sorted_pairs],
            [y for _, y in sorted_pairs],
            k=self.degree
        )
        return replace(
            self,
            knots=new_knots,
            coeffs=new_coeffs
        )

    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> BSplineApprox[X]:
        return self.update(xy_vals_seq)

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        if isinstance(other, BSplineApprox):
            return \
                np.all(np.abs(self.knots - other.knots) <= tolerance).item() \
                and \
                np.all(np.abs(self.coeffs - other.coeffs) <= tolerance).item()

        return False


@dataclass(frozen=True)
class AdamGradient:
    learning_rate: float
    decay1: float
    decay2: float

    @staticmethod
    def default_settings() -> AdamGradient:
        return AdamGradient(
            learning_rate=0.001,
            decay1=0.9,
            decay2=0.999
        )


@dataclass(frozen=True)
class Weights:
    adam_gradient: AdamGradient
    time: int
    weights: np.ndarray
    adam_cache1: np.ndarray
    adam_cache2: np.ndarray

    @staticmethod
    def create(
        weights: np.ndarray,
        adam_gradient: AdamGradient = AdamGradient.default_settings(),
        adam_cache1: Optional[np.ndarray] = None,
        adam_cache2: Optional[np.ndarray] = None
    ) -> Weights:
        return Weights(
            adam_gradient=adam_gradient,
            time=0,
            weights=weights,
            adam_cache1=np.zeros_like(
                weights
            ) if adam_cache1 is None else adam_cache1,
            adam_cache2=np.zeros_like(
                weights
            ) if adam_cache2 is None else adam_cache2
        )

    def update(self, gradient: np.ndarray) -> Weights:
        time: int = self.time + 1
        new_adam_cache1: np.ndarray = self.adam_gradient.decay1 * \
            self.adam_cache1 + (1 - self.adam_gradient.decay1) * gradient
        new_adam_cache2: np.ndarray = self.adam_gradient.decay2 * \
            self.adam_cache2 + (1 - self.adam_gradient.decay2) * gradient ** 2
        corrected_m: np.ndarray = new_adam_cache1 / \
            (1 - self.adam_gradient.decay1 ** time)
        corrected_v: np.ndarray = new_adam_cache2 / \
            (1 - self.adam_gradient.decay2 ** time)

        new_weights: np.ndarray = self.weights - \
            self.adam_gradient.learning_rate * corrected_m / \
            (np.sqrt(corrected_v) + SMALL_NUM)

        return replace(
            self,
            time=time,
            weights=new_weights,
            adam_cache1=new_adam_cache1,
            adam_cache2=new_adam_cache2,
        )

    def within(self, other: Weights, tolerance: float) -> bool:
        return np.all(np.abs(self.weights - other.weights) <= tolerance).item()


@dataclass(frozen=True)
class LinearFunctionApprox(FunctionApprox[X]):

    feature_functions: Sequence[Callable[[X], float]]
    regularization_coeff: float
    weights: Weights
    direct_solve: bool

    @staticmethod
    def create(
        feature_functions: Sequence[Callable[[X], float]],
        adam_gradient: AdamGradient = AdamGradient.default_settings(),
        regularization_coeff: float = 0.,
        weights: Optional[Weights] = None,
        direct_solve: bool = True
    ) -> LinearFunctionApprox[X]:
        return LinearFunctionApprox(
            feature_functions=feature_functions,
            regularization_coeff=regularization_coeff,
            weights=Weights.create(
                adam_gradient=adam_gradient,
                weights=np.zeros(len(feature_functions))
            ) if weights is None else weights,
            direct_solve=direct_solve
        )

    def get_feature_values(self, x_values_seq: Iterable[X]) -> np.ndarray:
        return np.array(
            [[f(x) for f in self.feature_functions] for x in x_values_seq]
        )

    def representational_gradient(self, x_value: X) -> LinearFunctionApprox[X]:
        return replace(
            self,
            weights=replace(
                self.weights,
                weights=np.array([f(x_value) for f in self.feature_functions])
            )
        )

    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        return np.dot(
            self.get_feature_values(x_values_seq),
            self.weights.weights
        )

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        if isinstance(other, LinearFunctionApprox):
            return self.weights.within(other.weights, tolerance)

        return False

    def regularized_loss_gradient(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> np.ndarray:
        x_vals, y_vals = zip(*xy_vals_seq)
        feature_vals: np.ndarray = self.get_feature_values(x_vals)
        diff: np.ndarray = np.dot(feature_vals, self.weights.weights) \
            - np.array(y_vals)
        return np.dot(feature_vals.T, diff) / len(diff) \
            + self.regularization_coeff * self.weights.weights

    def update(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> LinearFunctionApprox[X]:
        gradient: np.ndarray = self.regularized_loss_gradient(xy_vals_seq)
        new_weights: np.ndarray = self.weights.update(gradient)
        return replace(self, weights=new_weights)

    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> LinearFunctionApprox[X]:
        if self.direct_solve:
            x_vals, y_vals = zip(*xy_vals_seq)
            feature_vals: np.ndarray = self.get_feature_values(x_vals)
            feature_vals_T: np.ndarray = feature_vals.T
            left: np.ndarray = np.dot(feature_vals_T, feature_vals) \
                + feature_vals.shape[0] * self.regularization_coeff * \
                np.eye(len(self.weights.weights))
            right: np.ndarray = np.dot(feature_vals_T, y_vals)
            ret = replace(
                self,
                weights=Weights.create(
                    adam_gradient=self.weights.adam_gradient,
                    weights=np.dot(np.linalg.inv(left), right)
                )
            )
        else:
            tol: float = 1e-6 if error_tolerance is None else error_tolerance

            def done(
                a: LinearFunctionApprox[X],
                b: LinearFunctionApprox[X],
                tol: float = tol
            ) -> bool:
                return a.within(b, tol)

            ret = iterate.converged(
                self.iterate_updates(itertools.repeat(xy_vals_seq)),
                done=done
            )

        return ret


@dataclass(frozen=True)
class DNNSpec:
    neurons: Sequence[int]
    bias: bool
    hidden_activation: Callable[[np.ndarray], np.ndarray]
    hidden_activation_deriv: Callable[[np.ndarray], np.ndarray]
    output_activation: Callable[[np.ndarray], np.ndarray]
    output_activation_deriv: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class DNNApprox(FunctionApprox[X]):

    feature_functions: Sequence[Callable[[X], float]]
    dnn_spec: DNNSpec
    regularization_coeff: float
    weights: Sequence[Weights]

    @staticmethod
    def create(
        feature_functions: Sequence[Callable[[X], float]],
        dnn_spec: DNNSpec,
        adam_gradient: AdamGradient = AdamGradient.default_settings(),
        regularization_coeff: float = 0.,
        weights: Optional[Sequence[Weights]] = None
    ) -> DNNApprox[X]:
        if weights is None:
            inputs: Sequence[int] = [len(feature_functions)] + \
                [n + (1 if dnn_spec.bias else 0)
                 for i, n in enumerate(dnn_spec.neurons)]
            outputs: Sequence[int] = list(dnn_spec.neurons) + [1]
            wts = [Weights.create(
                weights=np.random.randn(output, inp) / np.sqrt(inp),
                adam_gradient=adam_gradient
            ) for inp, output in zip(inputs, outputs)]
        else:
            wts = weights

        return DNNApprox(
            feature_functions=feature_functions,
            dnn_spec=dnn_spec,
            regularization_coeff=regularization_coeff,
            weights=wts
        )

    def get_feature_values(self, x_values_seq: Iterable[X]) -> np.ndarray:
        return np.array(
            [[f(x) for f in self.feature_functions] for x in x_values_seq]
        )

    def forward_propagation(
        self,
        x_values_seq: Iterable[X]
    ) -> Sequence[np.ndarray]:
        """
        :param x_values_seq: a n-length-sequence of input points
        :return: list of length (L+2) where the first (L+1) values
                 each represent the 2-D input arrays (of size n x |I_l|),
                 for each of the (L+1) layers (L of which are hidden layers),
                 and the last value represents the output of the DNN (as a
                 1-D array of length n)
        """
        inp: np.ndarray = self.get_feature_values(x_values_seq)
        ret: List[np.ndarray] = [inp]
        for w in self.weights[:-1]:
            out: np.ndarray = self.dnn_spec.hidden_activation(
                np.dot(inp, w.weights.T)
            )
            if self.dnn_spec.bias:
                inp = np.insert(out, 0, 1., axis=1)
            else:
                inp = out
            ret.append(inp)
        ret.append(
            self.dnn_spec.output_activation(
                np.dot(inp, self.weights[-1].weights.T)
            )[:, 0]
        )
        return ret

    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        return self.forward_propagation(x_values_seq)[-1]

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        if isinstance(other, DNNApprox):
            return all(w1.within(w2, tolerance)
                       for w1, w2 in zip(self.weights, other.weights))
        else:
            return False

    def backward_propagation(
        self,
        fwd_prop: Sequence[np.ndarray],
        objective_derivative_output: Callable[[np.ndarray], np.ndarray]
    ) -> Sequence[np.ndarray]:
        """
        :param
        fwd_prop represents the result of forward propagation, a sequence
        of (L+1) 2-D np.ndarrays, followed by a 1-D np.ndarray for the output
        of the DNN.
        objective_derivative_output represents the derivative of the objective
        function with respect to the output of the DNN

        :return: list (of length L+1) of |O_l| x |I_l| 2-D arrays,
                 i.e., same as the type of self.weights.weights
        This function computes the gradient (with respect to weights) of
        cross-entropy loss where the output layer activation function
        is the canonical link function of the conditional distribution of y|x
        """
        layer_inputs: Sequence[np.ndarray] = fwd_prop[:-1]
        deriv: np.ndarray = objective_derivative_output(fwd_prop[-1]) * \
            self.dnn_spec.output_activation_deriv(fwd_prop[-1])
        deriv = deriv.reshape(1, -1)
        back_prop: List[np.ndarray] = [np.dot(deriv, layer_inputs[-1]) /
                                       deriv.shape[1]]
        # L is the number of hidden layers, n is the number of points
        # layer l deriv represents dObj/dS_l where S_l = I_l . weights_l
        # (S_l is the result of applying layer l without the activation func)
        for i in reversed(range(len(self.weights) - 1)):
            # deriv_l is a 2-D array of dimension |O_l| x n
            # The recursive formulation of deriv is as follows:
            # deriv_{l-1} = (weights_l^T inner deriv_l) haddamard g'(S_{l-1}),
            # which is ((|I_l| x |O_l|) inner (|O_l| x n)) haddamard
            # (|I_l| x n), which is (|I_l| x n) = (|O_{l-1}| x n)
            # Note: g'(S_{l-1}) is expressed as hidden layer activation
            # derivative as a function of O_{l-1} (=I_l).
            deriv = np.dot(self.weights[i + 1].weights.T, deriv) * \
                self.dnn_spec.hidden_activation_deriv(layer_inputs[i + 1].T)
            # If self.dnn_spec.bias is True, then I_l = O_{l-1} + 1, in which
            # case # the first row of the calculated deriv is removed to yield
            # a 2-D array of dimension |O_{l-1}| x n.
            if self.dnn_spec.bias:
                deriv = deriv[1:]
            # layer l gradient is deriv_l inner layer_inputs_l, which is
            # of dimension (|O_l| x n) inner (n x (|I_l|) = |O_l| x |I_l|
            back_prop.append(np.dot(deriv, layer_inputs[i]) / deriv.shape[1])
        return back_prop[::-1]

    def representational_gradient(self, x_value: X) -> DNNApprox[X]:
        return replace(
            self,
            weights=replace(
                self.weights,
                weights=self.backward_propagation(
                    fwd_prop=self.forward_propagation([x_value]),
                    objective_derivative_output=lambda arr: np.ones_like(arr)
                )
            )
        )

    def regularized_loss_gradient(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> Sequence[np.ndarray]:
        """
        :param pairs: list of pairs of n (x, y) points
        :return: list (of length L+1) of |O_l| x |I_l| 2-D array,
                 i.e., same as the type of self.weights.weights
        This function computes the regularized gradient (with respect to
        weights) of cross-entropy loss where the output layer activation
        function is the canonical link function of the conditional
        distribution of y|x
        """
        x_vals, y_vals = zip(*xy_vals_seq)
        fwd_prop: Sequence[np.ndarray] = self.forward_propagation(x_vals)

        def obj_deriv_output(out: np.ndarray) -> np.ndarray:
            return (out - np.array(y_vals)) / \
                self.dnn_spec.output_activation_deriv(out)

        return [x + self.regularization_coeff * self.weights[i].weights
                for i, x in enumerate(self.backward_propagation(
                    fwd_prop=fwd_prop,
                    objective_derivative_output=obj_deriv_output
                ))]

    def update(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> DNNApprox[X]:
        return replace(
            self,
            weights=[w.update(g) for w, g in zip(
                self.weights,
                self.regularized_loss_gradient(xy_vals_seq)
            )]
        )

    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> DNNApprox[X]:
        tol: float = 1e-6 if error_tolerance is None else error_tolerance

        def done(
            a: DNNApprox[X],
            b: DNNApprox[X],
            tol: float = tol
        ) -> bool:
            return a.within(b, tol)

        return iterate.converged(
            self.iterate_updates(itertools.repeat(xy_vals_seq)),
            done=done
        )


def learning_rate_schedule(
    initial_learning_rate: float,
    half_life: float,
    exponent: float
) -> Callable[[int], float]:
    def lr_func(n: int) -> float:
        return initial_learning_rate * (1 + (n - 1) / half_life) ** -exponent
    return lr_func


if __name__ == '__main__':

    from scipy.stats import norm
    from pprint import pprint

    alpha = 2.0
    beta_1 = 10.0
    beta_2 = 4.0
    beta_3 = -6.0
    beta = (beta_1, beta_2, beta_3)

    x_pts = np.arange(-10.0, 10.5, 0.5)
    y_pts = np.arange(-10.0, 10.5, 0.5)
    z_pts = np.arange(-10.0, 10.5, 0.5)
    pts: Sequence[Tuple[float, float, float]] = \
        [(x, y, z) for x in x_pts for y in y_pts for z in z_pts]

    def superv_func(pt):
        return alpha + np.dot(beta, pt)

    n = norm(loc=0., scale=2.)
    xy_vals_seq: Sequence[Tuple[Tuple[float, float, float], float]] = \
        [(x, superv_func(x) + n.rvs(size=1)[0]) for x in pts]

    ag = AdamGradient(
        learning_rate=0.5,
        decay1=0.9,
        decay2=0.999
    )
    ffs = [
        lambda _: 1.,
        lambda x: x[0],
        lambda x: x[1],
        lambda x: x[2]
    ]

    lfa = LinearFunctionApprox.create(
         feature_functions=ffs,
         adam_gradient=ag,
         regularization_coeff=0.001,
         direct_solve=True
    )

    lfa_ds = lfa.solve(xy_vals_seq)
    print("Direct Solve")
    pprint(lfa_ds.weights)
    errors: np.ndarray = lfa_ds.evaluate(pts) - \
        np.array([y for _, y in xy_vals_seq])
    print("Mean Squared Error")
    pprint(np.mean(errors * errors))
    print()

    print("Linear Gradient Solve")
    for _ in range(100):
        print("Weights")
        pprint(lfa.weights)
        errors: np.ndarray = lfa.evaluate(pts) - \
            np.array([y for _, y in xy_vals_seq])
        print("Mean Squared Error")
        pprint(np.mean(errors * errors))
        lfa = lfa.update(xy_vals_seq)
        print()

    ds = DNNSpec(
        neurons=[2],
        bias=True,
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda x: np.ones_like(x),
        output_activation=lambda x: x,
        output_activation_deriv=lambda x: np.ones_like(x)
    )

    dnna = DNNApprox.create(
        feature_functions=ffs,
        dnn_spec=ds,
        adam_gradient=ag,
        regularization_coeff=0.01
    )
    print("DNN Gradient Solve")
    for _ in range(100):
        print("Weights")
        pprint(dnna.weights)
        errors: np.ndarray = dnna.evaluate(pts) - \
            np.array([y for _, y in xy_vals_seq])
        print("Mean Squared Error")
        pprint(np.mean(errors * errors))
        dnna = dnna.update(xy_vals_seq)
        print()
