import matplotlib.pyplot as plt
import numpy as np


# Function to record mouse click event coordinates
def onclick(event):
    x = round(event.xdata, 2)
    y = round(event.ydata, 2)

    global median_coords
    global point_coords
    global med_count

    if med_count <= p:
        plt.scatter(x, y, color="blue")
        plt.draw()

        med_count += 1
        median_coords.append((x, y))
    else:
        plt.scatter(x, y, color="red")
        plt.draw()

        point_coords.append((x, y))


if __name__ == "__main__":
    # Number of medians
    p = 10
    med_count = 1
    
    # Array to store coordinates
    median_coords = []
    point_coords = []

    # Create a figure and define button event
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('button_press_event', onclick)

    # Set axis limits and show plot
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.show()

    with open('data\\toy_data3.txt', 'w+') as file:
        file.write(str(p) + ' ' + str(len(median_coords) + len(point_coords)) + '\n')

        for item in median_coords:
            file.write(str(item[0]) + ', ' + str(item[1]) + '\n')

        for item in point_coords:
            file.write(str(item[0]) + ', ' + str(item[1]) + '\n')
