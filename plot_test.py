import matplotlib.pyplot as plt
import numpy as np


# Function to record mouse click event coordinates
def onclick(event):
    x = round(event.xdata, 2)
    y = round(event.ydata, 2)
    plt.scatter(x, y, color="blue")
    plt.draw()
    coords.append((x, y))

# Array to store coordinates
coords = []

# Create a figure and define button event
fig, ax = plt.subplots()
fig.canvas.mpl_connect('button_press_event', onclick)

# Set axis limits and show plot
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.show()

print(coords)