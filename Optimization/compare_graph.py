import matplotlib.pyplot as plt
import numpy as np

def compareTwoAlg(cities,minAlg1,minAlg2,xtitle,ytitle,alg1name,alg2name):
    xpos = np.arange(len(cities))
    plt.title("Time per iter:" + alg1name +" vs "+alg2name)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    plt.bar(xpos + 0.2, minAlg1, width=0.4, label=alg1name, color='#88c999', alpha=0.7)
    plt.bar(xpos - 0.2, minAlg2, width=0.4, label=alg2name, color='cyan', alpha=0.7)
    plt.xticks(xpos, cities)
    plt.yscale('log')
    plt.legend()
    plt.show()

def plot_bar():
    aco = [91.85, 783.43,  2763.08, 8343.71]
    sa = [87.64, 768.78, 2830.10, 8647.73]
    compareTwoAlg([7, 25, 50, 100], aco, sa,"num of cities", "min distance", "ACO", "SA")

def plot_time_bar():
    aco = [0.007, 0.03,  0.118, 0.39]
    ga = [0.0000012, 0.000015, 0.000023, 0.000046]
    compareTwoAlg([7, 25, 50, 100], aco, ga,"num of cities", "time per iter", "ACO", "SA")

plot_time_bar()