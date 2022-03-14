import tsp_utils
from AntColony import ant_colony
from GeneticAlgorithm import geneticAlgorithm, City, geneticAlgorithmStats
import random
import math
import numpy as np
from queue import Queue
from SimulatedAnnealing import SimulatedAnnealing
from PlotAns import plot_path, plotTSP, plotTSPGA, PlotImprove
import Graph
from BranchAndBound import travelingSalesperson, tsp
import acopy
import math
import random
import networkx as nx
import matplotlib.pyplot as plt

class NodeGenerator:
    def __init__(self, width, height, nodesNumber):
        self.width = width
        self.height = height
        self.nodesNumber = nodesNumber

    def generate(self, coords):
        xs = np.array([coords[i][0] for i in range(len(coords))])
        ys = np.array([coords[i][1] for i in range(len(coords))])

        return np.column_stack((xs, ys))

def ACOInput(cities, size):
    coords = {}
    for i in range(cities):
        coords[i] = (random.random() * size, random.random() * size)
    return coords

def GAInput(cities, coords):
    cityList = []
    for i in range(cities):
        cityList.append(City(x=coords[i][0], y=coords[i][1]))
    return cityList

def SAInput(coords):
    return np.column_stack(coords)

def distance(start, end):
    x_distance = abs(start[0] - end[0])
    y_distance = abs(start[1] - end[1])
    return math.sqrt(pow(x_distance, 2) + pow(y_distance, 2))

def finalDist(te, ans, cities) :
    dist = 0
    for i in range(cities):
        if i != cities - 1:
            dist += distance(te[ans[i]], te[ans[i + 1]])
    return dist

def finalDistBAB(arr, cities):
    dist = 0
    for i in range(cities):
        if i != cities - 1:
            dist += distance(arr[i], arr[i + 1])
    return dist

def distanceGA(city1, city2):
    xDis = abs(city1.x - city2.x)
    yDis = abs(city1.y - city2.y)
    distance = np.sqrt((xDis ** 2) + (yDis ** 2))
    return distance

def finalDistGA(route, l_route) :
    dist = 0
    for i in range(l_route):
        if i != l_route - 1:
            dist += distanceGA(route[i], route[i + 1])
    return dist


def run_and_compare_only_avgSA(cities, size, num_iters):
    sumACO = 0
    sumSA = 0
    for i in range(num_iters):
        coords = ACOInput(cities, size)
        coordsSA = NodeGenerator(size, size, cities).generate(coords)
        colony = ant_colony(coords, distance, start=None, ant_count=50, alpha=1, beta=1.2,
                            pheromone_evaporation_coefficient=0.40, pheromone_constant=1000.0, iterations=500)
        answer = colony.mainloop()
        aco = finalDist(coords, answer, cities)
        sumACO += aco
        sa = SimulatedAnnealing(coordsSA, temp=900, alpha=0.9990, stopping_temp=0.00000001, stopping_iter=10000000)
        sumSA += sa.anneal()

    return [sumACO / num_iters, sumSA / num_iters]

def run_and_compare_only_avgGA(cities, size, num_iters):

    sumGA = 0
    sumACO = 0
    for i in range(num_iters):
        coords = ACOInput(cities, size)
        coordsGA = GAInput(cities, coords)
        colony = ant_colony(coords, distance, start=None, ant_count=50, alpha=1, beta=1.2,
                            pheromone_evaporation_coefficient=0.40, pheromone_constant=1000.0, iterations=500)
        answer = colony.mainloop()
        aco = finalDist(coords, answer, cities)
        sumACO += aco
        b_routeGA = geneticAlgorithm(population=coordsGA, popSize=100, eliteSize=20,
                                   mutationRate=0.001, generations=300)
        sumGA += finalDistGA(b_routeGA, len(b_routeGA))

    return [sumACO / num_iters, sumGA / num_iters]

def run_and_compareSA(cities, size, num_iters) :
    tAco = []
    tSa = []
    sumACO = 0
    sumSA = 0
    t = 0
    for i in range(num_iters):
        coords = ACOInput(cities, size)
        coordsSA = NodeGenerator(size, size, cities).generate(coords)
        # ACO
        # colony = ant_colony(coords, distance, start=None, ant_count=50, alpha=1, beta=1.2,
        #                     pheromone_evaporation_coefficient=0.40, pheromone_constant=1000.0, iterations=500)
        # answer = colony.mainloop()
        # aco = finalDist(coords, answer, cities)
        # sumACO += aco
        # tAco.append(round(aco))
        # SA
        sim_an = SimulatedAnnealing(coordsSA, temp=900, alpha=0.9990, stopping_temp=0.00000001, stopping_iter=10000000)
        sa = sim_an.anneal()
        sumSA += sa
        tSa.append(round(sa))
        times, ttime = sim_an.ret_times()
        t += ttime

    print("SA - avg time per iteration=", t / num_iters)
    print("ACO - all distances :")
    print(tAco)
    print("SA - all distances :")
    print(tSa)
    print("-----")
    print("ACO : Avg min={}".format(sumACO / num_iters))
    print("SA : Avg min={}".format(sumSA / num_iters))
    return [sumACO / num_iters, sumSA / num_iters]

def run_and_compareGA(cities, size, num_iters) :
    sumACO = 0
    sumGA = 0
    for i in range(num_iters):
        coords = ACOInput(cities, size)
        coordsGA = GAInput(cities, coords)
        # ACO
        colony = ant_colony(coords, distance, start=None, ant_count=50, alpha=1, beta=1.2,
                                pheromone_evaporation_coefficient=0.40, pheromone_constant=1000.0, iterations=500)
        answer = colony.mainloop()
        aco = finalDist(coords, answer, cities)
        sumACO += aco
        timeAC, ttimeACO = colony.ret_t_stats()
        # GA
        b_routeGA, histGA, initl, ga, timesGA, ttimeGA = geneticAlgorithmStats(population=coordsGA, popSize=100, eliteSize=20,
                                   mutationRate=0.001, generations=300)
        sumGA += ga

        print("times data ")
        print("ACO - avg time per iter ={}".format(timeAC))
        print("GA - avg time per iter ={}".format(ttimeGA))
        print("ACO - all distances :")
        print(colony.ret_stats())
        print("GA - all distances :")
        print(histGA)

    print("-----")
    print("ACO : Avg min={}".format(sumACO / num_iters))
    print("GA : Avg min={}".format(sumGA))
    return [sumACO / num_iters, sumGA / num_iters]


def input_from_file(f_name):
    coords = {}
    cs = []
    f = open(f_name, "r")
    for line in f:
        i, x, y = line.split()
        coords[int(i) - 1] = (float(x), float(y))
        cs.append((float(x), float(y)))
    f.close()
    return (coords, cs)


def compute_from_fileACO(f_name, num_iters, plot) :
    coords, cs = input_from_file(f_name)
    cities = len(coords)
    sumACO = 0
    for i in range(num_iters):
        colony = ant_colony(coords, distance, start=None, ant_count=50, alpha=1, beta=1.2,
                            pheromone_evaporation_coefficient=0.40, pheromone_constant=1000.0, iterations=500)
        answer = colony.mainloop()
        sumACO += finalDist(coords, answer, cities)
        initd = str(round(colony.stats[0]))
        print("ACO - avg time per iter=", colony.ttime)
        print("ACO - time per iteration")
        print(colony.times)
        if plot:
            PlotImprove(colony.stats, "Ant Colony Alg min distance per iter", "start dist={}, final dist={}".format(initd, str(round(sumACO))))
            plotTSP([answer], coords, "Ant Colony Alg Final Path. Dist={}".format(str(round(sumACO))), 1)
    return sumACO / num_iters


def compute_from_fileGA(f_name, num_iters, plot) :
    coords, cs = input_from_file(f_name)
    cities = len(coords)
    sumGA = 0
    for i in range(num_iters):
        coordsGA = GAInput(cities, coords)
        b_routeGA, hist, initd, finald, times, ttime = geneticAlgorithmStats(population=coordsGA, popSize=100, eliteSize=20,
                                     mutationRate=0.001, generations=3000)
        sumGA += finalDistGA(b_routeGA, len(b_routeGA))
        print("GA - avg time per iter=", ttime)
        print("GA - time per iteration")
        print(times)
        if plot:
            PlotImprove(hist, "Genetic Alg min distance per iter", "start dist={}, final dist={}".format(initd, finald))
            plotTSPGA(b_routeGA, "Genetic Alg Final Path. Dist={}".format(round(sumGA)), 1)
    return sumGA / num_iters


def compute_from_fileSA(f_name, num_iters, plot) :
    coords, cs = input_from_file(f_name)
    cities = len(coords)
    sumSA = 0
    for i in range(num_iters):
        coordsSA = NodeGenerator(1000, 1000, cities).generate(coords)
        sa = SimulatedAnnealing(coordsSA, temp=1000, alpha=0.9995, stopping_temp=0.00000001, stopping_iter=10000000)
        sumSA += round(sa.anneal())
        times, ttime = sa.ret_times()
        print("SA - avg time per iteration=", ttime)
        if plot:
            PlotImprove(sa.weight_list, "Simulated Annealing min distance per iter", "start dist={}, final dist={}".format(round(sa.initial_weight), sumSA))
            plotTSP([sa.best_solution], coords, "Simulated Annealing Alg Final Path. Dist={}".format(str(round(sumSA))), 1)
    return sumSA / num_iters


def find_best_ants(cities, size, num_iters) :
    sumACO1 = 0
    sumAC02 = 0
    sumAC03 = 0
    sumAC04 = 0
    sumAC05 = 0
    for i in range(num_iters):
        coords = ACOInput(cities, size)
        # ACO
        colony = ant_colony(coords, distance, start=None, ant_count=90, alpha=1, beta=1.2,
                            pheromone_evaporation_coefficient=0.40, pheromone_constant=1000.0, iterations=80)
        answer = colony.mainloop()
        sumACO1 += colony.shortest_distance
        colony = ant_colony(coords, distance, start=None, ant_count=90, alpha=1, beta=1.2,
                            pheromone_evaporation_coefficient=0.40, pheromone_constant=1000.0, iterations=90)
        answer = colony.mainloop()
        sumAC02 += colony.shortest_distance
        colony = ant_colony(coords, distance, start=None, ant_count=90, alpha=1, beta=1.2,
                            pheromone_evaporation_coefficient=0.40, pheromone_constant=1000.0, iterations=100)
        answer = colony.mainloop()
        sumAC03 += colony.shortest_distance
        colony = ant_colony(coords, distance, start=None, ant_count=90, alpha=1, beta=1.2,
                            pheromone_evaporation_coefficient=0.40, pheromone_constant=1000.0, iterations=150)
        answer = colony.mainloop()
        sumAC04 += colony.shortest_distance
        colony = ant_colony(coords, distance, start=None, ant_count=90, alpha=1, beta=1.2,
                            pheromone_evaporation_coefficient=0.40, pheromone_constant=1000.0, iterations=300)
        answer = colony.mainloop()
        sumAC05 += colony.shortest_distance

    print("ACO , 0.40: Avg min={}".format(sumACO1 / num_iters))
    print("ACO , 0.45: Avg min={}".format(sumAC02 / num_iters))
    print("ACO , 0.50: Avg min={}".format(sumAC03 / num_iters))
    print("ACO , 0.35: Avg min={}".format(sumAC04 / num_iters))
    print("ACO , 0.90: Avg min={}".format(sumAC05 / num_iters))

def run_and_compareBAB(cities, size, num_iters) :
    tGa = []
    tAco = []
    tSa = []
    tBab = []
    sumBAB = 0
    sumGA = 0
    sumACO = 0
    sumSA = 0
    for i in range(num_iters):
        coords = {}
        coordsBaB = []
        for j in range(cities):
            x1 = random.random() * size
            y1 = random.random() * size
            coords[j] = (x1, y1)
            coordsBaB.append((x1, y1))
        coordsGA = GAInput(cities, coords)
        coordsSA = NodeGenerator(size, size, cities).generate(coords)
        # BAB
        branch_a_bound = [[x, y] for x, y in coordsBaB]
        path = tsp(branch_a_bound)
        bab = finalDistBAB(path, cities)
        sumBAB += bab
        tBab.append(round(bab))
        # ACO
        colony = ant_colony(coords, distance, start=None, ant_count=50, alpha=1, beta=3,
                            pheromone_evaporation_coefficient=0.40, pheromone_constant=1000.0, iterations=500)
        answer = colony.mainloop()
        aco = colony.shortest_distance
        sumACO += aco
        tAco.append(round(aco))
        # GA
        b_routeGA = geneticAlgorithm(population=coordsGA, popSize=100, eliteSize=20,
                                   mutationRate=0.01, generations=500)
        ga = finalDistGA(b_routeGA, len(b_routeGA))
        sumGA += ga
        tGa.append(round(ga))
        # SA
        sim_an = SimulatedAnnealing(coordsSA, temp=900, alpha=0.9990, stopping_temp=0.00000001, stopping_iter=10000000)
        sa = sim_an.anneal()
        sumSA += sa
        tSa.append(round(sa))

    print("BAB - all distances :")
    print(tBab)
    print("ACO - all distances :")
    print(tAco)
    print("GA - all distances :")
    print(tGa)
    print("SA - all distances :")
    print(tSa)
    print("-----")
    print("BAB: Avg min={}".format(sumBAB / num_iters))
    print("ACO : Avg min={}".format(sumACO / num_iters))
    print("GA : Avg min={}".format(sumGA / num_iters))
    print("SA : Avg min={}".format(sumSA / num_iters))
    return [sumBAB / num_iters, sumACO / num_iters, sumGA / num_iters, sumSA / num_iters]

def euclidean(a, b):
    return math.sqrt(pow(a[1] - b[1], 2) + pow(a[0] - b[0], 2))

def testant(rho, q, alpha, beta, iter, ants):
    cities = 25
    size = 200
    solver = acopy.Solver(rho=rho, q=q)
    colony = acopy.Colony(alpha=alpha, beta=beta)
    colony.get_ants(ants)
    coords = []
    for j in range(cities):
        coords.append((random.random() * size, random.random() * size))
    G = nx.from_numpy_matrix(np.array(tsp_utils.vectorToDistMatrix(NodeGenerator(size, size, cities).generate(coords))))
    tour = solver.solve(G, colony, limit=iter)
    print(tour.path)
    return tour.cost

def scatter_BAB(bab, aco, ga, cities):

    fig, ax = plt.subplots()
    plt.title("Min distance per iteration")
    plt.xlabel("Iterations")
    plt.ylabel("Path length")
    y = np.array(bab)
    x = np.array([i for i in range(cities)])
    s = np.array([250 for _ in range(cities)])
    ax.scatter(x, y, c='hotpink', s=s, label='BaB', alpha=0.5, edgecolors='none')

    y = np.array(aco)
    ax.scatter(x, y, c='#88c999', s=s, label='ACO', alpha=0.5, edgecolors='none')

    y = np.array(ga)
    ax.scatter(x, y, c='cyan', s=s, label='GA', alpha=0.5, edgecolors='none')
    ax.legend()
    plt.show()


def comapreGA_ACO():
    print("----")
    print("comp GA - ACO ")
    print("cities: 7 - ")
    print(run_and_compareGA(7, 50, 1))
    print("cities: 25 - ")
    print(run_and_compareGA(25, 200, 1))
    print("cities: 50 - ")
    print(run_and_compareGA(50, 500, 1))
    print("cities: 100 - ")
    print(run_and_compareGA(100, 1000, 1))
    print("----")

def comapreSA_ACO():
    print("----")
    print("comp SA - ACO ")
    print("cities: 7 - ")
    print(run_and_compareSA(7, 50, 50))
    print("cities: 25 - ")
    print(run_and_compareSA(25, 200, 50))
    print("cities: 50 - ")
    print(run_and_compareSA(50, 500, 20))
    print("cities: 100 - ")
    print(run_and_compareSA(100, 1000, 10))
    print("----")


