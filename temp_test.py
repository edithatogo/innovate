
import numpy as np
from src.innovate.backends.numpy_backend import NumPyBackend
from src.innovate.backends.jax_backend import JaxBackend

# Create backend instances and test all functionality
numpy_backend = NumPyBackend()
jax_backend = JaxBackend()

# Test all methods we added documentation for
print('Testing NumPyBackend methods...')
result = numpy_backend.array([1, 2, 3])
result = numpy_backend.sum([1., 2., 3.])
result = numpy_backend.mean([1., 2., 3.])
result = numpy_backend.where(np.array([True, False, True]), [1, 2, 3], [4, 5, 6])
result = numpy_backend.zeros(5)
result = numpy_backend.ones(5)
result = numpy_backend.max([1., 2., 3.])
result = numpy_backend.median([1., 2., 3.])
result = numpy_backend.min([1., 2., 3.])
result = numpy_backend.exp([1., 2., 3.])
result = numpy_backend.any([True, False, False])
result = numpy_backend.all([True, True, False])
result = numpy_backend.squeeze(np.array([[[1]], [[2]], [[3]]]))
result = numpy_backend.repeat([1, 2, 3], 2)
result = numpy_backend.power([2., 3., 4.], 2)

# Test with axis parameter to cover both return paths
arr_2d = np.array([[1, 2], [3, 4]])
result = numpy_backend.sum(arr_2d, axis=0)  # Should return array
result = numpy_backend.sum(arr_2d)  # Should return scalar
result = numpy_backend.mean(arr_2d, axis=0)  # Should return array
result = numpy_backend.mean(arr_2d)  # Should return scalar
result = numpy_backend.max(arr_2d, axis=0)  # Should return array
result = numpy_backend.max(arr_2d)  # Should return scalar
result = numpy_backend.min(arr_2d, axis=0)  # Should return array
result = numpy_backend.min(arr_2d)  # Should return scalar
result = numpy_backend.median(arr_2d, axis=0)  # Should return array
result = numpy_backend.median(arr_2d)  # Should return scalar

# Also test other 'like' methods
template = np.array([[1, 2], [3, 4]])
result = numpy_backend.ones_like(template)
result = numpy_backend.zeros_like(template)
result = numpy_backend.empty_like(template)
result = numpy_backend.full_like(template, 5.0)

print('All NumPyBackend functionality tested.')

print('Testing JAXBackend methods...')
result = jax_backend.array([1, 2, 3])
result = jax_backend.exp([1., 2., 3.])
result = jax_backend.power([2., 3., 4.], 2)
result = jax_backend.sum([1., 2., 3.])
result = jax_backend.mean([1., 2., 3.])
result = jax_backend.max([1., 2., 3.])
result = jax_backend.median([1., 2., 3.])

# Test with axis parameter
arr_2d = jax_backend.array([[1, 2], [3, 4]])
result = jax_backend.sum(arr_2d, axis=0)  # Should return array
result = jax_backend.sum(arr_2d)  # Should return scalar
result = jax_backend.mean(arr_2d, axis=0)  # Should return array
result = jax_backend.mean(arr_2d)  # Should return scalar
result = jax_backend.max(arr_2d, axis=0)  # Should return array
result = jax_backend.max(arr_2d)  # Should return scalar
result = jax_backend.median(arr_2d, axis=0)  # Should return array
result = jax_backend.median(arr_2d)  # Should return scalar

print('All JAXBackend functionality tested.')
