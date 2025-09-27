from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    values: List[float] = list(vals)

    left: List[float] = list(values)
    left[arg] -= epsilon / 2

    right: List[float] = list(values)
    right[arg] += epsilon / 2

    return (f(*right) - f(*left)) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """

    visited: set[int] = set()
    result: List[Variable] = []

    def dfs(variable: Variable) -> None:
        visited.add(variable.unique_id)

        if not variable.is_constant():
            for parent in variable.parents:
                if parent.unique_id not in visited:
                    dfs(parent)

        result.append(variable)

    dfs(variable)
    return result[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    order: Iterable[Variable] = topological_sort(variable)
    derivs: dict[int, Any] = {}
    derivs[variable.unique_id] = deriv

    for var in order:
        if var.is_leaf() or var.is_constant():
            continue

        partial_derivs: Iterable[Tuple[Variable, Any]] = var.chain_rule(derivs[var.unique_id])
        for partial_deriv, value in partial_derivs:
            if partial_deriv.is_leaf():
                partial_deriv.accumulate_derivative(value)
            else:
                if partial_deriv.unique_id not in derivs:
                    derivs[partial_deriv.unique_id] = 0.0
                derivs[partial_deriv.unique_id] += value


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
