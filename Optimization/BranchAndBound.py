from random import random

from scipy.spatial.distance import squareform, pdist
from tsp_solver.greedy import solve_tsp
import Graph
import copy
from heapq import *

class TSPNode:
    '''
    This class represents a partial solution to the Traveling
    Salesperson Problem
    '''

    # Import a coumputeBound function and adds that as a behavior to
    # this class - allows us to easily try different computeBound
    # methods
    from computeBound import computeBound


    def __init__(self, aGraph, aPath=[], pathLength=0):
        '''
        Creates a new node with this state and bounds
        '''
        self.state = aGraph
        self.path = copy.copy(aPath)
        self.pathLength = pathLength

        # Compute the new bound for this node
        self.bound = self.computeBound()


    def addVertex(self, vertex):
        '''
        Add the given vertex to our path and compute a new bound for
        ourself
        '''

        # If we already have a start vertex, our length is the length
        # from the last node in the path to the new vertex.
        if len(self.path):

            self.pathLength += self.state.getAt(self.path[-1],vertex)
        else:
            self.pathLength = 0  # Redundant since it's done in init()
        # Append this new vertex to our path list
        self.path.append(vertex)
        # Compute a new bound for this node
        self.bound = self.computeBound()


    # Methods to implement relational operations between nodes based on
    # the bound - necessary so we can put nodes into a priority queue
    # and have them arranged in the proper order
    def __lt__(self, otherNode):
        return self.bound < otherNode.bound
    def __le__(self, otherNode):
        return self.bound <= otherNode.bound
    def __gt__(self, otherNode):
        return self.bound > otherNode.bound
    def __ge__(self, otherNode):
        return self.bound >= otherNode.bound




def travelingSalesperson(aGraph):
    '''
    Find an optimal tour, if one exists, for the given graph
    Arguments:
        aGraph - a Graph with nodes and edges
    Returns:
        (number of nodes visited, optimal tour TSPNode )
    '''

    visitedNodes = 0 # to keep track of how many nodes we 'visit'
    optimalTour = None # Until we discover the first tour

    # Use a list as a priority queue ordering nodes by their bounds
    priorityQueue = []

    # Starts with a root node where vertex 0 is the beginning of the
    # path and the path is empty with length 0
    currentNode = TSPNode(aGraph)

    # Add the first vertex to the existing node. This computes a new
    # bounds for this node
    currentNode.addVertex(aGraph.getNames()[0])

    # Prime the pump by pushing this initial TSPnode onto the priority
    # queue
    heappush(priorityQueue, currentNode)

    # While there are more nodes to explore, pop off the "best" node and
    # see if it's a solution. If so and it's better than the existing
    # best we remember it as the best. Otherwise, generate all of its
    # children by adding another vertex to the end of the path and
    # computing the bound and enqueueing the child
    while (len(priorityQueue) > 0):
        # Get the "best" node off the priority queue
        currentNode = heappop(priorityQueue)

        # If the bound on this node looks better than current best,
        # visit, otherwise we're done!
        if not optimalTour or (currentNode.bound < optimalTour.pathLength):
            visitedNodes += 1
            # Check to see if we have a solution by checking to see if
            # all the nodes are part of the tour represented by this
            # node
            if len(currentNode.path) == aGraph.size():
                # Looks like we have a tour!  See if there's a path back
                # from the last node in the path to the first node in
                # the path, if so, add that edge and see if this is
                # optimal
                if aGraph.getAt(currentNode.path[-1], currentNode.path[0]) != None:
                    # Complete the cyle by adding the edge back to our
                    # beginning node
                    currentNode.addVertex(currentNode.path[0])
                    # Find out if this tour is shorter than the current best tour
                    if not optimalTour or \
                          (currentNode.pathLength < optimalTour.pathLength):
                        # Yes it's better so it's our new optimal tour
                        optimalTour = currentNode
            else:
                # haven't found a tour yet, expand all children nodes by
                # taking the current path and adding every vertex to the
                # end where that vertex is 1) adjacent to the end and 2)
                # not already part of the path
                for node in aGraph.getNames():
                    # If the node underconsideration is not already in
                    # our path, then is there an edge from the end of the
                    # current path to the node under consideration?
                    if node not in currentNode.path and \
                          aGraph.getAt(currentNode.path[-1], node) != None:
                        # Add this node to the path by creating a new node with
                        # this node at the end of the path
                        newNode = TSPNode(currentNode.state,
                                          currentNode.path,
                                          currentNode.pathLength)
                        newNode.addVertex(node)
                        # Only enqueue this new node if its bound is better than
                        # the current best, otherwise we just drop it
                        if not optimalTour or \
                              (newNode.bound < optimalTour.pathLength):
                            heappush(priorityQueue, newNode)

        else:  # Nothing left in priority queue is better so we're done!
            return (visitedNodes, optimalTour)

    # We've explored all the nodes in the priority queue and found none
    # really better, so return the best we found.  Note that if we
    # didn't find ANY tour, then optimalTour will be None - an
    # appropriate value to return anyway
    return (visitedNodes, optimalTour)


def tsp(dist_array) :

    inp = pdist(dist_array)

    D = squareform(inp)
    p = solve_tsp(D)
    path = []
    for i in range(len(p)):
        path.append((dist_array[p[i]][0], dist_array[p[i]][1]))
    return path

if __name__ == '__main__':

    D = [[],
         [1.0],
         [2.0, 3.0]]
    dist_array = [[32, 107], [44, 94], [135, 121], [105, 54], [125, 26]]
    D = squareform(pdist(dist_array))

    path = solve_tsp(D)

    # will print [1,0,2], path with total length of 3.0 units
    print(path)

    # c = []
    # c2 = []
    # for i in range(15):
    #     x = random() * 200
    #     y = random() * 200
    #     c.append((i, x, y))
    #     c2.append((x, y))
    #
    # dist_array = [[32, 107], [44, 94], [135, 121], [105, 54], [125, 26]]
    # dist_matrix = squareform(pdist(dist_array))
    # print(dist_matrix)
    # for i in range(25):
    #     print(c[i][0], " ", c[i][1], " ", c[i][2])
    # print(" ")
    # print(c2)
    # Create a random graph with 7 vertices. 3 is the random seed for
    # populating the graph and 1.0 says it has 100% density so it's fully
    # connected - less dense graphs may or may not have a tour!
    # arr = [(13.348699454875756, 169.98131827951104), (10.36770478285205, 115.9188195925158), (155.65618460115851, 62.4081332410461), (134.97936281994959, 147.3831716611422),
    #        (163.480741036018, 177.08591938529642), (188.18844271440392, 158.67855844113978), (46.10484995681248, 177.60976713010743), (52.1794755829609, 31.341712035350234),
    #        (189.66485792350238, 24.006600486564224), (123.2772288636792, 135.4087444399992), (159.38335697348955, 129.98287834190882), (109.85385649152393, 40.90376582172468),
    #        (160.29496213426822, 71.83892310162618), (53.37898247437918, 154.67277439984227), (37.61805212780096, 93.27276446047755), (37.88596535382063, 121.5584377865558),
    #        (15.145910498845371, 32.858477762419035), (18.1649947888435, 24.884445754859197), (142.78415887044417, 145.18460405855492), (61.90911832761643, 30.393808422324554), (173.89510397634638, 141.5994899267693),
    #        (125.09103893329161, 168.80510244575444), (92.4315701498265, 159.87913345912315), (11.632760138639075, 92.08578881832277), (118.40827213031528, 48.000776842867786)]
    # aGraph = Graph.fromTSPFile("eil51.tsp")
    # print(aGraph)
    # # bGraph = Graph.fromArr(c2, 15)
    # # print(bGraph)
    #
    # # Run the Traveling Salesperson on the graph and find out how many
    # # nodes we visited
    # # (visited, solution) = travelingSalesperson(aGraph)
    # # print('Visisted ', visited, ' nodes')
    # # if solution:
    # #     print('Shortest tour is ',solution.pathLength, ' long:')
    # #     print(solution.path)
    # # else:
    # #     print('No tour found')
    #
    # (visited, solution) = travelingSalesperson(aGraph)
    # print('Visisted ', visited, ' nodes')
    # if solution:
    #     print('Shortest tour is ', solution.pathLength, ' long:')
    #     print(solution.path)
    # else:
    #     print('No tour found')

"""
BAB: Avg min=324.5182597848899
ACO : Avg min=220.43193549073072
GA : Avg min=220.57115785318615
SA : Avg min=324.51825978488984

"""