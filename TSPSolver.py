#!/usr/bin/python3
import random

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
    from PyQt6.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
from heapq import *
import itertools


def reduceMatrix(arr, lb):  # Function to reduce the given matrix and update the lower bound
    n = len(arr)
    for row in range(n):  # Check rows for minimum value
        minItem = np.min(arr[row])
        if minItem == float('inf'):  # this row has been used and can be ignored
            continue
        lb += minItem  # add the minimum to the lower bound (will often be 0)
        arr[row] -= minItem  # reduce the values in this row
    for col in range(n):  # Check columns for minimum value
        minItem = np.min(arr[:, col])
        if minItem == float('inf'):  # this row has been used and can be ignored
            continue
        lb += minItem  # add the minimum to the lower bound (will often be 0)
        arr[:, col] -= minItem  # reduce the values in this column
    return arr, lb


class BBsubProblem:  # An object to store a subinstance of the branch and bound problem
    def __init__(self, rcm, priority, lb, level, curPath, cityID):
        self.rcm = rcm
        self.priority = priority
        self.lb = lb
        self.level = level
        self.path = curPath
        self.cityId = cityID

    def getDepth(self):
        return len(self.path)

    def __gt__(self, other):  # overriden greater than function so that objects can be stored in priority queue
        if self.priority > other.priority:
            return True
        elif self.priority < other.priority:
            return False
        elif self.level >= other.level:  # if priority is the same, use level
            return True
        else:
            return False


def calcPriority(lb, level, ncities):  # calculate the priority for subproblems
    # if level > 0:
    #     adjustFactor = (10*ncities*level) * np.log(ncities/level) # Enhanced entropy function. Will return high values for levels in the middle of the total number of cities
    #     print("Using adjust %d for level %d with %d cities" % (adjustFactor, level, ncities))
    # else:
    #     adjustFactor = 0
    adjustFactor = 10000 * level  # todo test
    return lb - adjustFactor  # include level in priority so that higher level subProblems get precedence


def expandSubProb(subProb, source, dest):  # expand a problem with given source and destination
    newRCM = subProb.rcm.copy()
    newRCM[source] = float('inf')  # set row to inf
    newRCM[:, dest] = float('inf')  # set col to inf
    newLB = subProb.lb + subProb.rcm[source][dest]
    newRCM, newLB = reduceMatrix(newRCM, newLB)
    level = subProb.level + 1
    key = calcPriority(newLB, level, len(subProb.rcm))
    curPath = subProb.path + [source]  # build list of cities indexs that have been used
    newProb = BBsubProblem(newRCM, key, newLB, level, curPath, dest)
    return newProb


def initialRunsAlgo(ncities):  # Return how many times to run the random algorith for initial bssf. Depends on #cities
    if (ncities) < 20:
        return 5
    else:
        return 1


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution,
        time spent to find solution, number of permutations tried during search, the
        solution found, and three null values for fields not used for this
        algorithm</returns>
    '''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this
        algorithm</returns>
    '''

    def greedy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        rcm = np.full((ncities, ncities), float('inf'), dtype=float)  # todo faster if int
        # Fill cost matrix
        for i in range(ncities):
            for j in range(ncities):
                rcm[i][j] = cities[i].costTo(cities[j])  # TODO convert to int?

        while not foundTour and time.time() - start_time < time_allowance:
            newRCM = rcm.copy()
            curCity = random.randint(0, ncities - 1)  # get random starting city
            newRCM[:, curCity] = float(
                'inf')  # prevent any paths from returning here before all cities have been reached
            route = [curCity]
            for i in range(ncities):  # for each city
                nextCity = int(np.where(newRCM[curCity] == np.amin(newRCM[curCity]))[0][
                                   0])  # get index of minpath out of this city
                newRCM[:, nextCity] = float('inf')  # prevent any other incoming edges to this city
                route.append(nextCity)  # add to route
                curCity = nextCity  # update current city
            bssf = TSPSolution([cities[x] for x in route])  # convert to tsp solution
            count += 1
            if bssf.cost < np.inf:  # check if solution is valid
                # Found a valid route
                foundTour = True

        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints:
        max queue size, total number of states created, and number of pruned states.</returns>
    '''

    def branchAndBound(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        updatesToBSSF = 0
        numStatesCreated = 0
        numLeavesFound = 0
        numPruned = 0
        maxQueueSize = 1
        bssf = self.greedy().get("soln")
        for i in range(initialRunsAlgo(
                ncities)):  # Run a faster algorithm n times and take most optimal solution as initial bssf
            solution = self.greedy().get("soln")
            if solution.cost < bssf.cost:
                bssf = solution
        start_time = time.time()
        rcm = np.full((ncities, ncities), float('inf'), dtype=float)  # todo faster if int
        # Fill cost matrix
        for i in range(ncities):
            for j in range(ncities):
                rcm[i][j] = cities[i].costTo(cities[j])  # TODO convert to int?
        # Calculate reduced cost matrix
        lb = 0
        rcm, lb = reduceMatrix(rcm, lb)
        level = 0
        key = calcPriority(lb, level, ncities)  # includes the level in priority for better depth
        curPath = []
        root = BBsubProblem(rcm, key, lb, level, curPath, 0)  # start at city 0 always
        numStatesCreated += 1
        hq = []
        # Start priority queue
        heappush(hq, root)

        while len(hq) > 0 and time.time() - start_time < time_allowance:
            # Take top from queue
            maxQueueSize = max(maxQueueSize, len(hq))
            subProb = heappop(hq)
            # Prune? Final?
            if subProb.lb > bssf.cost:
                numPruned += 1
                continue  # Skip this subproblem

            # Expand into subproblems
            cityIndex = subProb.cityId
            for to in range(ncities):  # check all possible desitantions
                if subProb.rcm[cityIndex][
                    to] + subProb.lb > bssf.cost:  # ignore cities where path to is greater than current bssf
                    numPruned += 1
                    continue
                newProb = expandSubProb(subProb, cityIndex, to)  # create new subproblem
                numStatesCreated += 1
                if newProb.lb >= bssf.cost:
                    numPruned += 1  # pruned because solution had too high a lower bound after reduction
                    continue
                if newProb.getDepth() == ncities:  # if it could be possible solution (max depth)
                    numLeavesFound += 1

                    possibleSolution = TSPSolution([cities[x] for x in newProb.path])  # create the solution
                    cost = possibleSolution.cost
                    if cost != float('inf') and cost < bssf.cost:  # compare to bssf and update if better
                        bssf = possibleSolution
                        updatesToBSSF += 1
                else:
                    heappush(hq, newProb)  # push new problem onto queue

        # Return values
        end_time = time.time()
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = updatesToBSSF
        results['soln'] = bssf
        results['max'] = maxQueueSize
        results['total'] = numPruned + numStatesCreated
        for leftOverState in hq:  # include the number of unused states on the queue that would have been pruned
            if leftOverState.lb > bssf.cost:
                numPruned += 1
        results['pruned'] = numPruned
        return results

    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number of solutions found during search, the
        best solution found.  You may use the other three field however you like.
        algorithm</returns>
    '''

    def fancy(self, time_allowance=60.0, NUM_ANTS=100, PHEROMONE_WEIGHT=1.0, STANDARD_WEIGHT=2.0, WEIGHT_CONSTANT=0.01,
              CHECK_FOR_CONVERGENCE=100, distance_adjustment=100):

        # Functions
        def get_transition_probability(idx1, idx2):
            return pow(edge_weights[idx1][idx2], PHEROMONE_WEIGHT) * pow(edge_distances[idx1][idx2], -STANDARD_WEIGHT)

        def get_probablistic_path_from(source):
            path = []
            dist = 0.0
            path.append(source)
            curr_idx = source
            while len(path) < ncities:
                n_sum = 0.0
                possible_next = []
                for n in range(ncities):
                    if n in path or edge_distances[curr_idx][n] == float('inf'):  # already visited or inf
                        continue
                    n_sum += get_transition_probability(curr_idx, n)
                    possible_next.append(n)

                if len(possible_next) == 0:  # avoid getting caught when no more possible edges
                    return path, float('inf')

                r = np.random.uniform(0.0, n_sum)
                x = 0.0
                for nn in possible_next:
                    x += get_transition_probability(curr_idx, nn)
                    if r <= x:
                        dist += edge_distances[curr_idx][nn]
                        curr_idx = nn
                        path.append(nn)
                        break
            dist += edge_distances[curr_idx][source]
            return path, dist

        cities = self._scenario.getCities()
        ncities = len(cities)
        converged = False
        count = 0
        old_bssf = 1000000
        new_bssf = 1000000
        bssf_path = []
        start_time = time.time()
        edge_distances = np.full((ncities, ncities), float('inf'), dtype=float)
        # Fill weight matrix
        for i in range(ncities):
            for j in range(ncities):
                edge_distances[i][j] = cities[i].costTo(cities[j])
        edge_distances += distance_adjustment  # Todo this is so that the weight isn't inf
        edge_weights = np.full((ncities, ncities), 1.0, dtype=float)

        # Main loop
        while not converged and time.time() - start_time < time_allowance:
            for k in range(CHECK_FOR_CONVERGENCE):
                # Evaporation step
                edge_weights *= 0.999
                new_weights = edge_weights.copy()
                # run all ants
                for j in range(NUM_ANTS):

                    cost = float('inf')  # Get a valid path
                    while cost == float('inf'):
                        antpath, cost = get_probablistic_path_from(random.randint(0, ncities - 1))

                    if cost < new_bssf:
                        new_bssf = cost
                        bssf_path = antpath

                    diff = cost - new_bssf + 0.05  # avoid a difference of 0 for division by 0 error todo use new_bssf?
                    weight_update = WEIGHT_CONSTANT / diff

                    for i in range(ncities):  # todo needs + 1?
                        source = antpath[i % ncities]
                        dest = antpath[(i + 1) % ncities]
                        new_weights[source][dest] += weight_update  # update weight for edge in both directions
                        new_weights[dest][source] += weight_update
                # update edge weights and normalize to sum to 1
                for i in range(ncities):
                    rowsum = sum(new_weights[i])  # todo does this work?
                    for j in range(ncities):
                        # multiplying by 2 since every node has two neighbors eventually
                        edge_weights[i][j] = 2 * new_weights[i][j] / rowsum
            # check for converge
            if new_bssf == old_bssf and len(bssf_path) == ncities:
                converged = True
            old_bssf = new_bssf

        solution = TSPSolution([cities[x] for x in bssf_path])  # create the solution
        # if solution.cost != new_bssf:
        #     exit(-2) # This should never happen
        # Return values
        results = {}
        end_time = time.time()
        results['cost'] = solution.cost
        results['time'] = end_time - start_time
        results['count'] = None  # updatesToBSSF
        results['soln'] = solution
        results['max'] = None  # maxQueueSize
        results['total'] = None  # numPruned + numStatesCreated
        results['pruned'] = None  # numPruned
        return results
