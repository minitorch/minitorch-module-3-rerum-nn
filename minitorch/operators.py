"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(a: float, b: float) -> float:
    """Multiple two arguments.

    Args:
    ----
        a: first argument
        b: second argument

    Returns:
    -------
        a * b

    """
    return a * b


def id(a: float) -> float:
    """Identical func.

    Args:
    ----
        a: argument

    Returns:
    -------
        a

    """
    return a


def add(a: float, b: float) -> float:
    """Return the sum of two numbers.

    Args:
    ----
        a (float): First addend.
        b (float): Second addend.

    Returns:
    -------
        float: The sum a + b.

    """
    return a + b


def neg(a: float) -> float:
    """Return the additive inverse of a number.

    Args:
    ----
        a (float): Input value.

    Returns:
    -------
        float: The negation -a.

    """
    return -a


def lt(a: float, b: float) -> float:
    """Compare two numbers and return 1.0 if the first is strictly less than the second.

    Args:
    ----
        a (float): Left operand for comparison.
        b (float): Right operand for comparison.

    Returns:
    -------
        float: 1.0 if a < b, 0.0 otherwise.

    """
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    """Compare two numbers for exact equality and return 1.0 if they are equal.

    Args:
    ----
        a (float): Left operand for comparison.
        b (float): Right operand for comparison.

    Returns:
    -------
        float: 1.0 if a == b, 0.0 otherwise.

    """
    return 1.0 if a == b else 0.0


def max(a: float, b: float) -> float:
    """Return the larger of two numbers.

    Args:
    ----
        a (float): First value.
        b (float): Second value.

    Returns:
    -------
        float: The maximum of a and b.

    """
    return a if a > b else b


def is_close(a: float, b: float, eps: float = 1e-2) -> bool:
    """Check if two numbers are close within an absolute tolerance.

    The comparison is absolute-only: |a - b| < eps.

    Args:
    ----
        a (float): First value.
        b (float): Second value.
        eps (float, optional): Absolute tolerance. Defaults to 1e-2.

    Returns:
    -------
        bool: True if the values are within eps of each other, otherwise False.

    """
    return abs(a - b) < eps


def sigmoid(a: float) -> float:
    """Compute the logistic sigmoid function.

    Uses a numerically stable formulation:
        sigmoid(a) = 1 / (1 + exp(-a)) if a >= 0
                    exp(a) / (1 + exp(a)) if a < 0

    Args:
    ----
        a (float): Input value.

    Returns:
    -------
        float: A value in the open interval (0, 1).

    """
    return 1 / (1 + math.exp(-a)) if a >= 0 else math.exp(a) / (1 + math.exp(a))


def relu(a: float) -> float:
    """Apply the Rectified Linear Unit (ReLU) function.

    ReLU(a) = max(0, a)

    Args:
    ----
        a (float): Input value.

    Returns:
    -------
        float: 0 if a <= 0, otherwise a.

    """
    return a if a > 0 else 0


def log(a: float) -> float:
    """Compute the natural logarithm (base e).

    Args:
    ----
        a (float): Input value; must be positive.

    Returns:
    -------
        float: The natural logarithm of a.

    Raises:
    ------
        ValueError: If a <= 0.

    """
    return math.log(a)


def exp(a: float) -> float:
    """Compute the exponential function e**a.

    Args:
    ----
        a (float): Exponent.

    Returns:
    -------
        float: exp(a).

    """
    return math.exp(a)


def inv(a: float) -> float:
    """Compute the multiplicative inverse.

    Args:
    ----
        a (float): Input value.

    Returns:
    -------
        float: 1 / a.

    Raises:
    ------
        ZeroDivisionError: If a == 0.

    """
    return 1 / a


def log_back(a: float, b: float) -> float:
    """Backward pass for the natural logarithm.

    Given upstream gradient b and input a, returns the gradient dL/da where
    L is the loss and y = log(a). Since dy/da = 1/a, dL/da = b * (1/a) = b / a.

    Args:
    ----
        a (float): Original input to log; must be nonzero.
        b (float): Upstream gradient.

    Returns:
    -------
        float: The gradient with respect to a, b / a.

    Raises:
    ------
        ZeroDivisionError: If a == 0.

    """
    return b / a


def inv_back(a: float, b: float) -> float:
    """Backward pass for the multiplicative inverse.

    For y = 1/a, dy/da = -1/a^2. Given upstream gradient b,
    dL/da = b * (-1/a^2) = -b / a^2.

    Args:
    ----
        a (float): Original input to inv; must be nonzero.
        b (float): Upstream gradient.

    Returns:
    -------
        float: The gradient with respect to a, -b / (a ** 2).

    Raises:
    ------
        ZeroDivisionError: If a == 0.

    """
    return -b / (a**2)


def relu_back(a: float, b: float) -> float:
    """Backward pass for the ReLU function.

    For y = ReLU(a), dy/da = 1 if a > 0 else 0.
    Given upstream gradient b, dL/da = b if a > 0 else 0.

    Args:
    ----
        a (float): Original input to ReLU.
        b (float): Upstream gradient.

    Returns:
    -------
        float: The gradient with respect to a, b if a > 0 else 0.

    """
    return b if a > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(a: Iterable[float], f: Callable[[float], float]) -> Iterable[float]:
    """Apply a function to each element of an iterable.

    Args:
    ----
        a (Iterable): Input iterable.
        f (Callable): Function to apply to each element.

    Returns:
    -------
        Iterable: A new iterable with f applied to each element of a.

    """
    return [f(x) for x in a]


def zipWith(a: Iterable[float], b: Iterable[float], f: Callable[[float, float], float]) -> Iterable[float]:
    """Combine two iterables element-wise using a binary function.

    Args:
    ----
        a (Iterable): First iterable.
        b (Iterable): Second iterable.
        f (Callable): Binary function to combine elements.

    Returns:
    -------
        Iterable: A new iterable with f(a[i], b[i]) for each i.

    Note:
    ----
        Stops when the shorter iterable is exhausted.

    """
    i_a, i_b = iter(a), iter(b)
    res = []
    while True:
        try:
            res.append(f(next(i_a), next(i_b)))
        except StopIteration:
            break
    return res


def reduce(a: Iterable[float], start: float, f: Callable[[float, float], float]) -> float:
    """Reduce an iterable to a single value using a binary function.

    Args:
    ----
        a (Iterable[float]): Input iterable to reduce.
        start (float): Initial accumulator value.
        f (Callable): Binary function for reduction.

    Returns:
    -------
        float: The final accumulated value.

    """
    accumulator = start
    for i in a:
        accumulator = f(accumulator, i)

    return accumulator


def negList(a: Iterable[float]) -> Iterable[float]:
    """Negate each element in a list.

    Args:
    ----
        a (Iterable[float]): Input iterable of numbers.

    Returns:
    -------
        Iterable[float]: A new iterable with each element negated.

    """
    return map(a, lambda x: -x)


def addLists(a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
    """Add two lists element-wise.

    Args:
    ----
        a (Iterable[float]): First list of numbers.
        b (Iterable[float]): Second list of numbers.

    Returns:
    -------
        Iterable[float]: A new list with element-wise sums.

    Note:
    ----
        Stops when the shorter list is exhausted.

    """
    return zipWith(a, b, lambda x, y: x + y)


def sum(a: Iterable[float]) -> float:
    """Calculate the sum of all elements in an iterable.

    Args:
    ----
        a (Iterable[float]): Input iterable of numbers.

    Returns:
    -------
        float: The sum of all elements in a.

    """
    return reduce(a, 0, lambda x, y: x + y)


def prod(a: Iterable[float]) -> float:
    """Calculate the product of all elements in an iterable.

    Args:
    ----
        a (Iterable[float]): Input iterable of numbers.

    Returns:
    -------
        float: The product of all elements in a.

    """
    return reduce(a, 1, lambda x, y: x * y)
