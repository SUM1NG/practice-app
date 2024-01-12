import numpy as np
data = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
counts = np.histogram(data, bins=[0, 2, 4, 6])[0]
print(counts)