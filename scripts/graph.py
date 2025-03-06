import matplotlib.pyplot as plt
import numpy as np

########################################################################################################################
# For gimini
# Data
datasets = ['Baseline', 'Extended', 'Full','Replaced', 'No Doc']
rephrased = [0.75, 0.80, 0.78, 0.85, 0.82]
without_rephrasing = [0.21, 0.022, 0.01, 0.83, 0.80]

x = np.arange(len(datasets))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the bars
bars = ax.bar(x - width/2, rephrased, width, label='Rephrased', color='skyblue')
bars2 = ax.bar(x + width/2, without_rephrasing, width, label='Without Rephrasing', color='lightcoral')

# Adding lines
ax.plot(x, rephrased, color='blue', marker='o', label='Rephrased Line')
ax.plot(x, without_rephrasing, color='red', marker='s', label='Without Rephrasing Line')

# Labels, title, and legend
ax.set_xlabel('Datasets')
ax.set_ylabel('Performance')
ax.set_title('Performance on Different Datasets with and without Rephrasing')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()

# Show the plot
plt.show()