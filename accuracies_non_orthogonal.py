from typing import Optional

# A relative half-width requirement or the accuracy parameter. Target value
# for the ratio of halfwidth to sample mean. If n_variables > 1,
# relative_accuracy can be a scalar to be used for all variables or a 1darray
# of values of size n_variables.
# For non-orthogonal cells, we can use a relative accuracy for all (non-zero) variables.
RELATIVE_ACCURACY: float = 0.01
ABSOLUTE_ACCURACY: Optional[float] = None
