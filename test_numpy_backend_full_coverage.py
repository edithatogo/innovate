import numpy as np
from src.innovate.backends.numpy_backend import NumPyBackend

# Create backend instance and test all method variations to get 100% coverage
backend = NumPyBackend()

# Test the mean method with where parameter specifically (covers lines 34-36)
arr = np.array([1, 2, 3, 4, 5])
where_condition = np.array([True, False, True, False, True])
result = backend.mean(arr, where=where_condition)
print(f'mean with where parameter: {result}')

# Test the solve_ode method with different inputs (covers line 56)
def simple_ode(y, t): 
    return -0.5 * y  # Different ODE
result = backend.solve_ode(simple_ode, [2.0], [0, 0.5, 1.0])
print(f'solve_ode result shape: {result.shape}')

# Test other methods with various parameters to cover edge cases
result = backend.sum([1, 2, 3, 4, 5], initial=10)
print(f'sum with initial: {result}')

condition = np.array([True, False, True])
result = backend.where(condition, [1, 2, 3], [4, 5, 6])
print(f'where result: {result}')

result = backend.diff([1, 4, 9, 16], n=2)
print(f'diff with n=2: {result}')  

result = backend.logsumexp([1, 2, 3], axis=0)
print(f'logsumexp with axis: {result}')

arrays = [np.array([1, 2]), np.array([3, 4])]
result = backend.stack(arrays)
print(f'stack result: {result}')

result = backend.matmul(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
print(f'matmul result: {result}')

result = backend.zeros((2, 3))
print(f'zeros with tuple shape: {result.shape}')

result = backend.ones((3, 2))
print(f'ones with tuple shape: {result.shape}')

result = backend.max([1, 5, 3])
print(f'max result: {result}')

result = backend.median([1, 5, 3, 7])
print(f'median result: {result}')

xp = np.array([0, 1, 2, 3, 4])
fp = np.array([0, 1, 4, 9, 16])
result = backend.interp(np.array([2.5]), xp, fp)
print(f'interp result: {result}')

# Test the new methods we added
result = backend.exp([1, 2, 3])
print(f'exp result: {result}')

result = backend.any([True, False, False])
print(f'any result: {result}')

result = backend.all([True, True, False])
print(f'all result: {result}')

arr = np.array([[[1], [2], [3]]])
result = backend.squeeze(arr)
print(f'squeeze result: {result}')

result = backend.repeat([1, 2, 3], 2)
print(f'repeat result: {result}')

# Test all other methods
result = backend.array([1, 2, 3])
print(f'array result: {result}')

result = backend.mean([1, 2, 3], axis=0)
print(f'mean result: {result}')

result = backend.sqrt([4, 9, 16])
print(f'sqrt result: {result}')

print('All tests completed successfully')