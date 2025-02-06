import numpy as np
from sklearn import gaussian_process
# Function to calculate the tanimoto similarity for the Gaussian process kernel prediction
def tanimoto_similarity(a, b):
    """Computes the Tanimoto similarity for all pairs.

  Args:
    a: Numpy array with shape [batch_size_a, num_features].
    b: Numpy array with shape [batch_size_b, num_features].

  Returns:
    Numpy array with shape [batch_size_a, batch_size_b].
  """
    aa = np.sum(a, axis=1, keepdims=True)
    bb = np.sum(b, axis=1, keepdims=True)
    ab = np.matmul(a, b.T)
    return np.true_divide(ab, aa + bb.T - ab)


class TanimotoKernel(gaussian_process.kernels.NormalizedKernelMixin,
                     gaussian_process.kernels.StationaryKernelMixin,
                     gaussian_process.kernels.Kernel):
  """Custom Gaussian process kernel that computes Tanimoto similarity."""

  def __init__(self):
    """Initializer."""
    pass  # Does nothing; this is required by get_params().

  def __call__(self, X, Y=None, eval_gradient=False):  # pylint: disable=invalid-name
    """Computes the pairwise Tanimoto similarity.

    Args:
      X: Numpy array with shape [batch_size_a, num_features].
      Y: Numpy array with shape [batch_size_b, num_features]. If None, X is
        used.
      eval_gradient: Whether to compute the gradient.

    Returns:
      Numpy array with shape [batch_size_a, batch_size_b].

    Raises:
      NotImplementedError: If eval_gradient is True.
    """
    if eval_gradient:
      raise NotImplementedError
    if Y is None:
      Y = X
    return tanimoto_similarity(X, Y)
  