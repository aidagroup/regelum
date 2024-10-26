"""Contains models.

These can be used in system dynamics fitting, critic and other tasks.

Updates to come.

"""

from copy import deepcopy
from scipy.stats import truncnorm
import regelum


from .utils import rg
from .typing import RgArray
try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor, FloatTensor
    from torch.nn import Module
    from torch.distributions.multivariate_normal import MultivariateNormal
    from torch.distributions.normal import Normal
except ModuleNotFoundError:
    from unittest.mock import MagicMock

    torch = MagicMock()
    nn = MagicMock()
    F = MagicMock()
    MultivariateNormal = MagicMock()

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Any, Tuple, Dict, Callable
import numpy as np
import casadi as cs
from omegaconf.listconfig import ListConfig
import math


def force_positive_def(func):
    def positive_def_wrapper(obj, *args, **kwargs):
        if obj.force_positive_def:
            return rg.soft_abs(func(obj, *args, **kwargs))
        else:
            return func(obj, *args, **kwargs)

    return positive_def_wrapper


def unversal_model_call(obj, *argin, weights=None, use_stored_weights=False):
    """_summary_

    Args:
        obj: an object from which the function is called
        weights: if passed, the model will try to perform forward with passed weights. Defaults to None.
        use_stored_weights: If True, computes forward using cached model (for Torch always with detached weights). Defaults to False.

    Returns:
        An array or scalar, depends on the model definition
    """
    if len(argin) == 2:
        left, right = argin
        if len(left.shape) != len(right.shape):
            raise ValueError(
                f"In {obj.__class__.__name__}.__call__ left and right arguments must have same number of dimensions!"
            )

        dim = len(left.shape)

        if dim == 1:
            argin = rg.concatenate(argin, axis=0)
        elif dim == 2:
            argin = rg.concatenate(argin, axis=1)
        else:
            raise ValueError(
                f"Wrong number of dimensions in {obj.__class__.__name__}.__call__"
            )
    elif len(argin) == 1:
        argin = argin[0]
    else:
        raise ValueError(
            f"Wrong number of arguments in {obj.__class__.__name__}.__call__. Can be either 1 or 2. Got: {len(argin)}"
        )

    if use_stored_weights is False:
        if weights is not None:
            result = obj.forward(argin, weights)
        else:
            result = obj.forward(argin)
    else:
        result = obj.cache.forward(argin)

    return result


class Model(regelum.RegelumBase, ABC):
    """Blueprint of a model."""

    def __call__(self, *argin, weights=None, use_stored_weights=False):
        return unversal_model_call(
            self, *argin, weights=weights, use_stored_weights=use_stored_weights
        )

    @property
    def weights(self):
        return self._weights

    @property
    def named_parameters(self):
        return self.weights

    @weights.setter
    def weights(self, new_weights):
        assert (
            self.weights.shape == new_weights.shape
        ), "The shape of weights was changed "
        f"in runtime from {self.weights.shape} to {new_weights.shape}"

        self._weights = new_weights

    @abstractmethod
    def __init__(self):
        """Initialize an instance of a model."""
        pass

    @abstractmethod
    def forward(self):
        pass

    def update_weights(self, weights):
        self.weights = weights

    def cache_weights(self, weights=None):
        if "cache" not in self.__dict__.keys():
            self.cache = deepcopy(self)

        if weights is None:
            self.cache.update_weights(self.weights)
        else:
            self.cache.update_weights(weights)

    def update_and_cache_weights(self, weights):
        self.cache_weights(weights)
        self.update_weights(weights)

    def restore_weights(self):
        """Assign the weights of the cached model to the active model.

        This may be needed when pytorch optimizer resulted in unsatisfactory weights, for instance.

        """
        self.update_and_cache_weights(self.cache.weights)


class ModelQuadLin(Model):
    """Base class for generic quadratic linear models.

    Normally used for running objective specification (diagonal quadratic matrix without linear terms) and critic polynomial models.
    """

    def __init__(
        self,
        quad_matrix_type: str,
        is_with_linear_terms: bool = False,
        dim_inputs: int = None,
        weights: np.array = None,
        weight_min: float = 1.0e-6,
        weight_max: float = 1.0e3,
        add_random_init_noise: bool = False,
    ):
        """Initialize an instance of quadratic-linear model.

        Args:
            quad_matrix_type (str): Type of quadratic matrix. Can be
                'diagonal', 'full' or 'symmetric'.
            is_with_linear_terms (bool, optional): Whether include
                linear terms or not, defaults to False
            dim_inputs (int, optional): Dimension of system's (agent's)
                inputs, defaults to None
            weights (_type_, optional): Manual set of model weights,
                defaults to None
            weight_min (float, optional): Lower bound for weights,
                defaults to 1.0e-6
            weight_max (float, optional): Upper bound for weights,
                defaults to 1.0e3
        """
        assert (
            dim_inputs is not None or weights is not None
        ), "Need dim_inputs or weights"

        self.quad_matrix_type = quad_matrix_type
        self.is_with_linear_terms = is_with_linear_terms
        self.single_weight_min = weight_min
        self.single_weight_max = weight_max

        if weights is None:
            self._calculate_dims(dim_inputs)
            self.weight_min = weight_min * rg.ones(self.dim_weights)
            self.weight_max = weight_max * rg.ones(self.dim_weights)
            self.weights = (self.weight_min + self.weight_max) / 20.0
        else:
            self._calculate_dims(self._calculate_dim_inputs(len(weights)))
            assert self.dim_weights == len(weights), "Wrong shape of dim_weights"
            self.weights = rg.array(weights)

        self.add_random_init_noise = add_random_init_noise
        if self.add_random_init_noise:
            self.weight_min = np.random.uniform(
                self.weight_min, self.weight_max, self.dim_weights
            )
            self.weight_max = np.random.uniform(
                self.weight_min, self.weight_max, self.dim_weights
            )
            self.weights = np.random.uniform(
                self.weight_min, self.weight_max, self.dim_weights
            )

        self.cache_weights(self.weights)

    def get_quad_lin(self, new_weights):
        if self.quad_matrix_type == "full":
            quad_matrix = rg.reshape(
                new_weights[: self.dim_quad], (self.dim_inputs, self.dim_inputs)
            )
        elif self.quad_matrix_type == "diagonal":
            quad_matrix = rg.diag(new_weights[: self.dim_quad])
        elif self.quad_matrix_type == "symmetric":
            quad_matrix = ModelQuadLin.quad_matrix_from_flat_weights(
                new_weights[: self.dim_quad]
            )

        linear_coefs = (
            new_weights[self.dim_quad :] if self.is_with_linear_terms else None
        )

        return quad_matrix, linear_coefs

    @Model.weights.setter
    def weights(self, new_weights):
        if new_weights is not None:
            self._weights = new_weights
            self._quad_matrix, self._linear_coefs = self.get_quad_lin(new_weights)

    @property
    def weight_bounds(self):
        return rg.array([[self.single_weight_min, self.single_weight_max]])

    def _calculate_dim_inputs(self, dim_weights):
        if self.quad_matrix_type == "diagonal":
            if self.is_with_linear_terms:
                return dim_weights // 2
            else:
                return dim_weights
        elif self.quad_matrix_type == "full":
            if self.is_with_linear_terms:
                return round((np.sqrt(1 + 4 * dim_weights) - 1) / 2)
            else:
                return round(np.sqrt(dim_weights))
        elif self.quad_matrix_type == "symmetric":
            if self.is_with_linear_terms:
                return round((np.sqrt(9 + 8 * dim_weights) - 1) / 2)
            else:
                return round((np.sqrt(1 + 8 * dim_weights) - 1) / 2)

    def _calculate_dims(self, dim_inputs):
        self.dim_inputs = dim_inputs
        self.dim_linear = dim_inputs if self.is_with_linear_terms else 0
        if self.quad_matrix_type == "diagonal":
            self.dim_quad = dim_inputs
        elif self.quad_matrix_type == "full":
            self.dim_quad = dim_inputs * dim_inputs
        elif self.quad_matrix_type == "symmetric":
            self.dim_quad = dim_inputs * (dim_inputs + 1) // 2

        self.dim_weights = self.dim_quad + self.dim_linear

    def cast_to_inputs_type(self, value, inputs):
        if value is None:
            return None
        if isinstance(inputs, torch.Tensor):
            device = inputs.device
            if not isinstance(value, torch.Tensor):
                return torch.FloatTensor(value).to(device)
            elif device != value.device:
                return value.to(device)
        if isinstance(inputs, cs.MX) and isinstance(value, np.ndarray):
            value = rg.DM(value)
        return value

    def forward(self, inputs, weights=None):
        if weights is None:
            weights = self.weights
            quad_matrix = self._quad_matrix
            linear_coefs = self._linear_coefs
            if isinstance(inputs, torch.Tensor):
                quad_matrix = torch.FloatTensor(quad_matrix).to(inputs.device)
                if linear_coefs is not None:
                    linear_coefs = torch.FloatTensor(linear_coefs).to(inputs.device)
        else:
            quad_matrix, linear_coefs = self.get_quad_lin(weights)

        return ModelQuadLin.quadratic_linear_form(
            inputs,
            quad_matrix,
            linear_coefs,
        )

    @staticmethod
    def quad_matrix_from_flat_weights(
        flat_weights: RgArray, tol=1e-7
    ):
        len_flat_weights = flat_weights.shape[0]
        dim_quad_matrix_float = (np.sqrt(1 + 8 * len_flat_weights) - 1) / 2
        dim_quad_matrix = round(dim_quad_matrix_float)
        assert np.isclose(
            dim_quad_matrix_float, dim_quad_matrix, tol
        ), f"Can't build quad matrix with flat_weights of dim {len_flat_weights}"

        quad_matrix = rg.zeros(
            (dim_quad_matrix, dim_quad_matrix), prototype=flat_weights
        )
        left_ids, right_ids = np.triu_indices(dim_quad_matrix)
        for weigth_idx, (i, j) in enumerate(zip(left_ids, right_ids)):
            quad_matrix[i, j] = flat_weights[weigth_idx]

        return quad_matrix

    @staticmethod
    def quadratic_linear_form(inputs, quad_matrix, linear_coefs=None):
        initial_dim_inputs = len(inputs.shape)
        assert (
            initial_dim_inputs == 1 or initial_dim_inputs == 2
        ), "Wrong shape of inputs can be 1d or 2d. Got {}".format(initial_dim_inputs)

        if initial_dim_inputs == 1:
            inputs = inputs.reshape(1, -1)
        assert (
            len(quad_matrix.shape) == 2
        ), "Wrong shape of quad matrix. Should be 2d. Got{}".format(
            len(quad_matrix.shape)
        )
        assert (
            quad_matrix.shape[0] == quad_matrix.shape[1]
        ), "Quad matrix should be square"
        assert (
            quad_matrix.shape[0] == inputs.shape[1]
        ), "Quad matrix should have same number of rows as inputs"

        quadratic_term = inputs @ quad_matrix @ inputs.T
        if len(quadratic_term.shape) > 0:
            quadratic_term = rg.diag(quadratic_term)

        if linear_coefs is not None:
            # assert (
            #     len(linear_coefs.shape) == 2 and linear_coefs.shape[0] == 1
            # ), "Wrong shape of linear coefs. Should be (1,n). Got {}".format(
            #     linear_coefs.shape
            # )

            # assert (
            #     quad_matrix.shape[1] == linear_coefs.shape[1]
            # ), "Quad matrix should have same number of columns as linear coefs"

            linear_term = inputs @ linear_coefs
            output = quadratic_term + linear_term
        else:
            output = quadratic_term

        if initial_dim_inputs == 1:
            output = output.reshape(-1)
        return output


class ModelWeightContainer(Model):
    """Trivial model, which is typically used in policy in which actions are being optimized directly."""

    def __init__(self, dim_output: int, weights_init: Optional[np.array] = None):
        """Initialize an instance of a model returns weights on call independent of input.

        Args:
            dim_output: output dimension
            weights_init: initial weights to set
        """
        self.dim_output = dim_output
        self._weights = weights_init
        self.weights_init = weights_init
        self.update_and_cache_weights(self._weights)

    def forward(self, *argin, weights=None):
        if weights is not None:
            return rg.force_row(weights[0, : self.dim_output])
        else:
            return rg.force_row(self._weights[0, : self.dim_output])


class ModelNN(nn.Module):
    """Class of pytorch neural network models. This class is not to be used barebones.

    Instead, you should inherit from it and specify your concrete architecture.

    """

    @property
    def cache(self):
        """Isolate parameters of cached model from the current model."""
        return self.cached_model[0]

    def detach_weights(self):
        """Excludes the model's weights from the pytorch computation graph.

        This is needed to exclude the weights from the decision variables in optimization problems.
        An example is temporal-difference optimization, where the old critic is to be treated as a frozen model.

        """
        for variable in self.parameters():
            variable.detach_()

    def cache_weights(self, whatever=None):
        """Assign the active model weights to the cached model followed by a detach.

        This method also backs up itself and performs this operation only once upon the initialization procedure
        """
        if "cached_model" not in self.__dict__.keys():
            self.cached_model = (
                deepcopy(self),
            )  ## this is needed to prevent cached_model's parameters to be parsed by Torch module init hooks

        self.cache.load_state_dict(self.state_dict())
        self.cache.detach_weights()

    @property
    def weights(self):
        return self.state_dict()

    def update_weights(self, whatever=None):
        pass

    def update_and_cache_weights(self, weights=None):
        if weights is not None:
            for item in weights:
                weights[item].requires_grad_()
            weights = self.load_state_dict(weights)
        self.cache_weights()

    def restore_weights(self):
        """Assign the weights of the cached model to the active model.

        This may be needed when pytorch optimizer resulted in unsatisfactory weights, for instance.

        """
        self.update_and_cache_weights(self.cache.state_dict())

    def __call__(self, *argin, weights=None, use_stored_weights=False):
        return unversal_model_call(
            self, *argin, weights=weights, use_stored_weights=use_stored_weights
        )


class WeightClipper:
    """Weight clipper for pytorch layers."""

    def __init__(
        self, weight_min: Optional[float] = None, weight_max: Optional[float] = None
    ):
        """Initialize a weight clipper.

        Args:
            weight_min: minimum value for weight
            weight_max: maximum value for weight
        """
        self.weight_min = weight_min
        self.weight_max = weight_max

    def __call__(self, module):
        if self.weight_min is not None or self.weight_max is not None:
            # filter the variables to get the ones you want
            module.weight.data.clamp_(self.weight_min, self.weight_max)


class ModelPerceptron(ModelNN):
    """Helper class to ease the creation of perceptron models."""

    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        dim_hidden: int,
        n_hidden_layers: int,
        hidden_activation: Optional[Module] = None,
        output_activation: Optional[Module] = None,
        force_positive_def: bool = False,
        is_force_infinitesimal: bool = False,
        is_bias: bool = True,
        weight_max: Optional[float] = None,
        weight_min: Optional[float] = None,
        linear_weights_init: Optional[Callable[[Tensor], Tensor]] = None,
        linear_weights_init_kwargs: Optional[Dict[str, Any]] = None,
        biases_init: Optional[Callable[[Tensor], Tensor]] = None,
        biases_init_kwargs: Optional[Dict[str, Any]] = None,
        output_bounds: Union[List[List[float]], np.array] = None,
        weights=None,
    ):
        """Initialize an instance of a fully-connected model.

        Args:
            dim_input (int): The dimensionality of the input.
            dim_output (int): The dimensionality of the output.
            dim_hidden (int): The dimensionality of the hidden linear
                layers (dim_hidden * dim_hidden).
            n_hidden_layers (int): The number of hidden layers.
            hidden_activation (Optional[nn.Module]): The activation
                function for the hidden layers. (Optional, defaults to
                None). If None then nn.LeakyReLU(0.2) is used.
            output_activation (Optional[nn.Module]): The activation
                function for the output layer. (Optional, defaults to
                None)
            force_positive_def (bool): Whether to force perceptron to be
                positive definite (if True rg.softabs is applied to output). (Optional,
                defaults to False)
            is_force_infinitesimal (bool): Whether to force perceptron
                to be equal to 0 with zero input. (Optional, defaults to
                False)
            is_bias (bool): Whether to include bias terms. (Optional,
                defaults to True)
            weight_max (Optional[float]): The maximum value for the
                weights. (Optional, defaults to None)
            weight_min (Optional[float]): The minimum value for the
                weights. (Optional, defaults to None)
            linear_weights_init (Optional[Callable[[torch.Tensor], torch.Tensor]]):
                The weight initialization function for the linear
                layers. (Optional, defaults to None)
            linear_weights_init_kwargs (Optional[Dict[str, Any]]):
                Additional keyword arguments for the linear weight
                initialization function. (Optional, defaults to None)
            biases_init (Optional[Callable[[torch.Tensor], torch.Tensor]]):
                The bias initialization function for the linear layers.
                (Optional, defaults to None)
            biases_init_kwargs (Optional[Dict[str, Any]]): Additional
                keyword arguments for the bias initialization function.
                (Optional, defaults to None)
            output_bounds (Union[List[List[float]], np.array]): The
                bounds for the output values. (Optional, defaults to
                None)
            weights (Optional): Pre-trained weights for the model.
                (Optional, defaults to None)
        """
        ModelNN.__init__(self)
        self.weight_clipper = WeightClipper(weight_min, weight_max)
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.n_hidden_layers = n_hidden_layers
        self.force_positive_def = force_positive_def
        self.is_force_infinitesimal = is_force_infinitesimal
        self.is_bias = is_bias
        if isinstance(dim_hidden, int):
            self.input_layer = nn.Linear(dim_input, dim_hidden, bias=is_bias)
            self.hidden_layers = nn.ModuleList(
                [
                    nn.Linear(dim_hidden, dim_hidden, bias=is_bias)
                    for _ in range(n_hidden_layers)
                ]
            )
        elif isinstance(dim_hidden, list) or isinstance(dim_hidden, ListConfig):
            assert (
                len(dim_hidden) == n_hidden_layers
            ), "number of passed hidden dimensions is not equal to n_hidden_layers"
            self.input_layer = nn.Linear(dim_input, dim_hidden[0], bias=is_bias)
            dim_hidden_full = zip(dim_hidden[:-1], dim_hidden[1:])
            self.hidden_layers = nn.ModuleList(
                [
                    nn.Linear(dim_hidden_first, dim_hidden_second, bias=is_bias)
                    for dim_hidden_first, dim_hidden_second in dim_hidden_full
                ]
            )
            dim_hidden = dim_hidden[-1]

        else:
            raise ValueError("inappropriate type of dim_hidden argument passed.")
        self.output_layer = nn.Linear(dim_hidden, dim_output, bias=is_bias)
        self.hidden_activation = (
            hidden_activation if hidden_activation is not None else nn.LeakyReLU(0.15)
        )
        self.output_activation = output_activation
        self.bounds_handler = (
            BoundsHandler(output_bounds) if output_bounds is not None else None
        )
        if linear_weights_init is not None or biases_init is not None:
            for layer in (
                [self.input_layer]
                + [hidden_layer for hidden_layer in self.hidden_layers]
                + [self.output_layer]
            ):
                if linear_weights_init is not None:
                    if linear_weights_init_kwargs is not None:
                        linear_weights_init(layer.weight, **linear_weights_init_kwargs)
                    else:
                        linear_weights_init(layer.weight)

                if biases_init is not None:
                    if biases_init_kwargs is not None:
                        biases_init(layer.bias, **biases_init_kwargs)
                    else:
                        biases_init(layer.bias)

        if weights is not None:
            self.load_state_dict(weights)

        self.cache_weights()

    def _forward(self, x: FloatTensor) -> FloatTensor:
        self.input_layer.apply(self.weight_clipper)
        x = self.hidden_activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            hidden_layer.apply(self.weight_clipper)
            x = self.hidden_activation(hidden_layer(x))
        x = self.output_layer(x)
        if self.output_activation is not None:
            x = self.output_activation(x)

        if self.bounds_handler is not None:
            x = self.bounds_handler.unscale_from_minus_one_one_to_bounds(torch.tanh(x))

        return x

    @force_positive_def
    def forward(
        self, input_tensor: FloatTensor, weights=None
    ) -> FloatTensor:
        if self.is_force_infinitesimal:
            return self._forward(input_tensor) - self._forward(
                torch.zeros_like(input_tensor)
            )

        return self._forward(input_tensor)


class ModelWeightContainerTorch(ModelNN):
    """Pytorch model that have raw weights as its parameter.

    When using output bounds, the `output_bounding_type` parameter determines how the output is bounded:
    - If `output_bounding_type` is set to "clip", the output is clipped to the specified bounds.
    - If `output_bounding_type` is set to "tanh", the output is scaled using the hyperbolic tangent function to fit within the specified bounds. (same as squash_output in stable baselines)
    """

    def __init__(
        self,
        dim_weights: Union[int, Tuple[int, int]],
        output_bounds: Optional[List[Any]] = None,
        output_bounding_type: str = "clip",
    ):
        """Instantiate ModelWeightContainerTorch.

        Args:
            dim_weights (Union[int, Tuple[int, int]]): The
                dimensionality of the weights.
            output_bounds (Optional[List[Any]]): Optional bounds of the
                output. If `None`, the output is not bounded. Defaults
                to None.
            output_bounding_type (str): The type of output bounding.
                Must be either "clip" or "tanh". Defaults to "clip".
        """
        assert (
            output_bounding_type == "clip" or output_bounding_type == "tanh"
        ), "output_bounding_type must be 'clip' or 'tanh'"

        ModelNN.__init__(self)
        self.bounds_handler = (
            BoundsHandler(output_bounds) if output_bounds is not None else None
        )
        self.output_bounding_type = output_bounding_type

        self.dim_weights = (
            (1, dim_weights) if isinstance(dim_weights, int) else dim_weights
        )
        self._weights = torch.nn.Parameter(
            torch.FloatTensor(torch.zeros(self.dim_weights)),
            requires_grad=True,
        )

        self.cache_weights()

    def forward(self, inputs, weights=None):
        if self.bounds_handler is not None:
            if self.output_bounding_type == "clip":
                with torch.no_grad():
                    self._weights.clip_(-1.0, 1.0)

        if len(inputs.shape) == 1:
            inputs_like = self._weights[0, :]
        elif len(inputs.shape) == 2:
            if inputs.shape[0] <= self.dim_weights[0]:
                inputs_like = self._weights[: inputs.shape[0], :]
            else:
                raise ValueError(
                    "ModelWeightContainerTorch: Wrong inputs shape! inputs.shape[0]"
                    f"(Got: {inputs.shape[0]}) should be <= dim_weights[0] (Got: {self.dim_weights[0]})."
                )
        else:
            raise ValueError("Wrong inputs shape! Can be either 1 or 2")

        if self.bounds_handler is not None:
            if self.output_bounding_type == "clip":
                # inputs_like are already clipped in the beggining of the function via WeightClipper
                return self.bounds_handler.unscale_from_minus_one_one_to_bounds(
                    inputs_like
                )
            elif self.output_bounding_type == "tanh":
                return self.bounds_handler.unscale_from_minus_one_one_to_bounds(
                    torch.tanh(inputs_like)
                )
        else:
            return inputs_like


class BoundsHandler(ModelNN):
    r"""Output layer that restricts the model's output within specified bounds. The formula used is: F^{-1}(y), where F represents a linear transformation from the given bounds to the range [-1, 1].

    It is recommended to stack this layer after applying the tanh activation function to the model's output.
    """

    def __init__(self, bounds: Union[List[List[float]], np.array], is_unscale=True):
        """Initialize a new instance of the BoundsHandler class.

        Args:
            bounds (Union[List[List[float]], np.array] The bounds should be provided as a 2-column array-like object. The first column represents the left bounds, and the second column represents the right bounds.):
                Bounds of the model's output.
            is_unscale (bool, optional): Flag indicating whether to
                unscale (from [-1, 1] to bounds) or scale (from bounds
                to [-1, 1]) in forward. Defaults to True.
        """
        ModelNN.__init__(self)
        self.is_unscale = is_unscale
        self.bounds = torch.nn.Parameter(
            torch.FloatTensor(bounds),
            requires_grad=False,
        )

    def forward(self, x):
        if self.is_unscale:
            return self.unscale_from_minus_one_one_to_bounds(x)
        else:
            return self.scale_from_bounds_to_minus_one_one(x)

    def get_unscale_coefs_from_minus_one_one_to_bounds(
        self,
    ) -> Tuple[FloatTensor, FloatTensor]:
        unscale_bias, unscale_multiplier = (
            self.bounds.mean(dim=1),
            (self.bounds[:, 1] - self.bounds[:, 0]) / 2.0,
        )
        return unscale_bias, unscale_multiplier

    def unscale_from_minus_one_one_to_bounds(
        self, x: FloatTensor
    ) -> FloatTensor:
        (
            unscale_bias,
            unscale_multiplier,
        ) = self.get_unscale_coefs_from_minus_one_one_to_bounds()

        return x * unscale_multiplier + unscale_bias

    def scale_from_bounds_to_minus_one_one(self, y):
        (
            unscale_bias,
            unscale_multiplier,
        ) = self.get_unscale_coefs_from_minus_one_one_to_bounds()

        return (y - unscale_bias) / unscale_multiplier


class MultiplyByConstant(nn.Module):
    """Represents a module that multiplies the input tensor by a constant value."""

    def __init__(self, constant: float) -> None:
        """Instatiate MultiplyByConstant.

        Args:
            constant (float): The constant value to multiply the input
                by.

        Returns:
            torch.Tensor: The tensor resulting from multiplying the
            input by the constant value.
        """
        super().__init__()
        self.constant = constant

    def forward(self, input: FloatTensor) -> FloatTensor:
        return input * self.constant


class PerceptronWithTruncatedNormalNoise(ModelPerceptron):
    """Represents a perceptron model that applies a forward pass to input data and adds normal noise to the output.

    The `PerceptronWithTruncatedNormalNoise` class provides the following functionality:

    - Sampling : After the forward pass, normal noise is added to the output of the perceptron. The standard deviations of the noise can be specified using the `stds` parameter. All this functionality is implemented in the forward method.
    - Truncation to output bounds: The `is_truncated_to_output_bounds` flag determines whether the noise should be truncated to the output bounds. When set to `True`, the noise values are generated from corresponding truncated normal distribution.
    """

    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        dim_hidden: int,
        n_hidden_layers: int,
        stds: Union[List[float], np.array] = None,
        hidden_activation: Optional[Module] = None,
        output_activation: Optional[Module] = None,
        force_positive_def: bool = False,
        is_force_infinitesimal: bool = False,
        is_bias: bool = True,
        weight_max: Optional[float] = None,
        weight_min: Optional[float] = None,
        linear_weights_init: Optional[Callable[[Tensor], Tensor]] = None,
        linear_weights_init_kwargs: Optional[Dict[str, Any]] = None,
        biases_init: Optional[Callable[[Tensor], Tensor]] = None,
        biases_init_kwargs: Optional[Dict[str, Any]] = None,
        output_bounds: Union[List[List[float]], np.array] = None,
        is_truncated_to_output_bounds: Optional[Dict[str, Any]] = False,
        weights=None,
    ):
        """Instantiate PerceptronWithTruncatedNormalNoise.

        Args:
            dim_input (int): The dimensionality of the input.
            dim_output (int): The dimensionality of the output.
            dim_hidden (int): The dimensionality of the hidden linear
                layers (dim_hidden * dim_hidden).
            n_hidden_layers (int): The number of hidden layers.
            hidden_activation (Optional[nn.Module]): The activation
                function for the hidden layers. If None,
                nn.LeakyReLU(0.2) is used. (Optional, defaults to None)
            output_activation (Optional[nn.Module]): The activation
                function for the output layer. (Optional, defaults to
                None)
            force_positive_def (bool): Whether to force the perceptron
                to be positive definite (if True, softabs is used).
                (Optional, defaults to False)
            is_force_infinitesimal (bool): Whether to force the
                perceptron to be equal to 0 with zero input. (Optional,
                defaults to False)
            is_bias (bool): Whether to include bias terms. (Optional,
                defaults to True)
            weight_max (Optional[float]): The maximum value for the
                weights. (Optional, defaults to None)
            weight_min (Optional[float]): The minimum value for the
                weights. (Optional, defaults to None)
            linear_weights_init (Optional[Callable[[torch.Tensor], torch.Tensor]]):
                The weight initialization function for the linear
                layers. (Optional, defaults to None)
            linear_weights_init_kwargs (Optional[Dict[str, Any]]):
                Additional keyword arguments for the linear weight
                initialization function. (Optional, defaults to None)
            biases_init (Optional[Callable[[torch.Tensor], torch.Tensor]]):
                The bias initialization function for the linear layers.
                (Optional, defaults to None)
            biases_init_kwargs (Optional[Dict[str, Any]]): Additional
                keyword arguments for the bias initialization function.
                (Optional, defaults to None)
            output_bounds (Union[List[List[float]], np.array]): The
                bounds for the output values. (Optional, defaults to
                None)
            weights (Optional): Pre-trained weights for the model.
                (Optional, defaults to None)
            stds (Optional[np.array]): The standard deviations for
                sampling the normal distribution. (Optional, defaults to
                None)
            is_truncated_to_output_bounds (bool): Whether to truncate
                the samples to the output bounds. (Optional, defaults to
                False)
        """
        super().__init__(
            dim_input=dim_input,
            dim_output=dim_output,
            dim_hidden=dim_hidden,
            n_hidden_layers=n_hidden_layers,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            force_positive_def=force_positive_def,
            is_force_infinitesimal=is_force_infinitesimal,
            is_bias=is_bias,
            weight_min=weight_min,
            weight_max=weight_max,
            output_bounds=output_bounds,
            linear_weights_init=linear_weights_init,
            linear_weights_init_kwargs=linear_weights_init_kwargs,
            biases_init=biases_init,
            biases_init_kwargs=biases_init_kwargs,
            weights=weights,
        )
        self.is_truncated_to_output_bounds = is_truncated_to_output_bounds
        self.output_bounds_array = np.array(output_bounds)
        if self.is_truncated_to_output_bounds:
            if output_bounds is None:
                raise AssertionError(
                    "Cannot truncate the output without specifying output bounds. Please set output_bounds and use is_truncated_to_output_bounds=True."
                )

        self.stds_array = np.array(stds).reshape(-1)
        assert (
            len(self.stds_array) == dim_output
        ), "The length of standard deviations should be equal to dim_output."
        self.stds = torch.nn.Parameter(
            torch.FloatTensor(self.stds_array), requires_grad=False
        )

    def log_pdf(self, distribution_params_input, log_prob_args):
        means = super().forward(distribution_params_input)
        normal = Normal(loc=means, scale=torch.ones_like(means) * self.stds)
        if self.is_truncated_to_output_bounds:
            left_bounds, right_bounds = (
                self.bounds_handler.bounds[:, 0],
                self.bounds_handler.bounds[:, 1],
            )
            return (
                normal.log_prob(log_prob_args)
                - torch.log(
                    normal.cdf(torch.ones_like(means) * right_bounds)
                    - normal.cdf(torch.ones_like(means) * left_bounds)
                )
            ).sum(axis=1)
        else:
            return normal.log_prob(log_prob_args).sum(axis=1)

    def forward(self, observation, is_means_only=False):
        mean = super().forward(observation)
        if is_means_only:
            return mean
        mean_numpy = mean.detach().cpu().numpy()
        if self.is_truncated_to_output_bounds:
            left_bounds, right_bounds = (
                self.output_bounds_array[None, :, 0],
                self.output_bounds_array[None, :, 1],
            )
            sampled_action = torch.FloatTensor(
                truncnorm(
                    loc=mean_numpy,
                    scale=np.ones_like(mean_numpy) * self.stds_array,
                    b=(right_bounds - mean_numpy) / self.stds_array,
                    a=(left_bounds - mean_numpy) / self.stds_array,
                )
                .rvs()
                .reshape(1, -1)
            )
        else:
            sampled_action = Normal(
                loc=mean, scale=self.stds * torch.ones_like(mean)
            ).sample()

        return sampled_action


class GaussianMeanStd(ModelNN):
    def __init__(
        self,
        mean: ModelNN,
        std: ModelNN,
        output_bounds: List[Union[List[float], np.ndarray]],
        is_truncated_to_output_bounds: bool = True,
    ):
        super().__init__()
        self.mean = mean
        self.std = std
        self.is_truncated_to_output_bounds = is_truncated_to_output_bounds
        self.bounds_handler = BoundsHandler(output_bounds)
        self.output_bounds_array = np.array(output_bounds)

        if is_truncated_to_output_bounds and output_bounds is None:
            raise ValueError(
                "output_bounds must be specified when is_truncated_to_output_bounds is True"
            )

    def forward(self, observation, is_means_only=False):
        mean, std = self.mean(observation), self.std(observation)
        # mean, std = self.mean(observation), self.std * torch.linalg.norm(observation)

        if is_means_only:
            return mean

        if self.is_truncated_to_output_bounds:
            mean_numpy = mean.detach().cpu().numpy()
            std_numpy = std.detach().cpu().numpy()
            left_bounds, right_bounds = (
                self.output_bounds_array[None, :, 0],
                self.output_bounds_array[None, :, 1],
            )
            sampled_action = torch.FloatTensor(
                truncnorm(
                    loc=mean_numpy,
                    scale=std_numpy,
                    b=(right_bounds - mean_numpy) / std_numpy,
                    a=(left_bounds - mean_numpy) / std_numpy,
                )
                .rvs()
                .reshape(1, -1)
            )
        else:
            sampled_action = Normal(loc=mean, scale=std).sample()

        return sampled_action

    def log_pdf(self, distribution_params_input, log_prob_args):
        means, stds = self.mean(distribution_params_input), self.std(
            distribution_params_input
        )
        normal = Normal(loc=means, scale=stds)
        if self.is_truncated_to_output_bounds:
            left_bounds, right_bounds = (
                self.bounds_handler.bounds[:, 0],
                self.bounds_handler.bounds[:, 1],
            )
            return (
                normal.log_prob(log_prob_args)
                - torch.log(
                    normal.cdf(torch.ones_like(means) * right_bounds)
                    - normal.cdf(torch.ones_like(means) * left_bounds)
                )
            ).sum(axis=1)
        else:
            return normal.log_prob(log_prob_args).sum(axis=1)

    def entropy(self, distribution_params_input):
        means, stds = self.mean(distribution_params_input), self.std(
            distribution_params_input
        )

        a = self.bounds_handler.bounds[:, 0] * torch.ones_like(means)
        b = self.bounds_handler.bounds[:, 1] * torch.ones_like(means)
        alpha = (a - means) / stds
        beta = (b - means) / stds

        standard_normal = Normal(
            loc=torch.zeros_like(means), scale=torch.ones_like(means)
        )
        z = standard_normal.cdf(beta) - standard_normal.cdf(alpha)

        phi_alpha = torch.exp(standard_normal.log_prob(alpha))
        phi_beta = torch.exp(standard_normal.log_prob(beta))

        return (
            torch.log(math.sqrt(2 * math.pi * math.e) * stds * z)
            + (alpha * phi_alpha - beta * phi_beta) / (2 * z)
        ).sum(axis=1)
