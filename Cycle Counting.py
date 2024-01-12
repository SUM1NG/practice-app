import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rainflow(data):
    stack = []
    cycles = []

    for point in data:
        while len(stack) >= 2:
            if (min(stack[-2], point) < stack[-1] < max(stack[-2], point)):
                cycles.append((stack[-2], stack[-1], point))
                stack.pop()
            else:
                break

        stack.append(point)

    while len(stack) >= 2:
        cycles.append((stack[-2], stack[-1]))
        stack.pop()

    return cycles

# Generate random data
data = np.random.rand(1000)

# Apply Rainflow algorithm
cycles = rainflow(data)

# Convert the cycles to a DataFrame
df = pd.DataFrame(cycles, columns=['Start', 'Peak', 'End'])

# Calculate the range and mean of each cycle
df['Range'] = df['Peak'] - df[['Start', 'End']].min(axis=1)
df['Mean'] = df[['Start', 'Peak', 'End']].mean(axis=1)

# Plot the range and mean of each cycle
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(df['Range'], bins=20, alpha=0.7, color='g')
plt.title('Cycle Range')
plt.xlabel('Range')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(df['Mean'], bins=20, alpha=0.7, color='b')
plt.title('Cycle Mean')
plt.xlabel('Mean')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


