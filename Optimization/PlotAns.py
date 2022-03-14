import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plot_path(coords):
    print(coords)
    df_dict = pd.DataFrame.from_dict(coords, orient='index', columns=['x', 'y'])
    print(df_dict)
    # df = pd.DataFrame(df_dict, columns=["x", "y"])
    # print(df)
    sns.relplot(x="x", y="y", markers=True, sort=False, ci="sd",
            dashes=False, kind="line", data=df_dict)
    plt.show()


import matplotlib.pyplot as plt


def plotTSP(path, points,  title, num_iters=1):
    """
    path: List of lists with the different orders in which the nodes are visited
    points: coordinates for the different nodes
    num_iters: number of paths that are in the path list

    """

    # Unpack the primary TSP path and transform it into a list of ordered
    # coordinates

    fig = plt.figure()
    fig.set_size_inches(25.5, 10.5, forward=True)

    x = []
    y = []
    for i in path[0]:
        x.append(points[i][0])
        y.append(points[i][1])

    plt.plot(x, y, 'co', label=title)
    plt.title(title, fontsize = 40)
    # Set a scale for the arrow heads (there should be a reasonable default for this, WTF?)
    a_scale = float(max(x)) / float(100)

    # Draw the older paths, if provided
    if num_iters > 1:

        for i in range(1, num_iters):

            # Transform the old paths into a list of coordinates
            xi = [];
            yi = [];
            for j in path[i]:
                xi.append(points[j][0])
                yi.append(points[j][1])

            plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]),
                      head_width=0, color='r',
                      length_includes_head=True, ls='dashed',
                      width=0.001 / float(num_iters))
            for i in range(0, len(x) - 1):
                plt.arrow(xi[i], yi[i], (xi[i + 1] - xi[i]), (yi[i + 1] - yi[i]),
                          head_width=a_scale, color='r', length_includes_head=0,
                          ls='dashed', width=0.001 / float(num_iters))

    # Draw the primary path for the TSP problem
    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width=0,
              color='g', length_includes_head=True)
    for i in range(0, len(x) - 1):
        plt.arrow(x[i], y[i], (x[i + 1] - x[i]), (y[i + 1] - y[i]), head_width=0,
                  color='g', length_includes_head=True)

    # Set axis too slitghtly larger than the set of x and y
    plt.xticks([])
    plt.yticks([])
    # plt.xlim(-2, max(x) * 1.1)
    # plt.ylim(-2, max(y) * 1.1)

    plt.show()

def plotTSPGA(points,  title, num_iters=1):
    """
    path: List of lists with the different orders in which the nodes are visited
    points: coordinates for the different nodes
    num_iters: number of paths that are in the path list

    """

    # Unpack the primary TSP path and transform it into a list of ordered
    # coordinates

    fig = plt.figure()
    fig.set_size_inches(25.5, 10.5, forward=True)

    x = []
    y = []
    for p in points:
        x.append(p.x)
        y.append(p.y)

    plt.plot(x, y, 'co', label=title)
    plt.title(title)
    # Set a scale for the arrow heads (there should be a reasonable default for this, WTF?)
    a_scale = float(max(x)) / float(100)

    # Draw the older paths, if provided
    # if num_iters > 1:
    #
    #     for i in range(1, num_iters):
    #
    #         # Transform the old paths into a list of coordinates
    #         xi = [];
    #         yi = [];
    #         for j in path[i]:
    #             xi.append(points[j][0])
    #             yi.append(points[j][1])
    #
    #         plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]),
    #                   head_width=0, color='r',
    #                   length_includes_head=True, ls='dashed',
    #                   width=0.001 / float(num_iters))
    #         for i in range(0, len(x) - 1):
    #             plt.arrow(xi[i], yi[i], (xi[i + 1] - xi[i]), (yi[i + 1] - yi[i]),
    #                       head_width=a_scale, color='r', length_includes_head=0,
    #                       ls='dashed', width=0.001 / float(num_iters))

    # Draw the primary path for the TSP problem
    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width=0,
              color='g', length_includes_head=True)
    for i in range(0, len(x) - 1):
        plt.arrow(x[i], y[i], (x[i + 1] - x[i]), (y[i + 1] - y[i]), head_width=0,
                  color='g', length_includes_head=True)

    # Set axis too slitghtly larger than the set of x and y
    plt.xticks([])
    plt.yticks([])
    plt.xlim(-2, max(x) * 1.1)
    plt.ylim(-2, max(y) * 1.1)

    plt.show()

def PlotImprove(arr, title, subt) :
    # print(arr)
    plt.suptitle(title, fontsize=18)
    plt.title(subt)
    plt.plot(arr)
    plt.show()