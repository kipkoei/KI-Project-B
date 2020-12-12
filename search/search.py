# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class Edge:
    def __init__(self, source, target, action, cost):
        self.source = source
        self.target = target
        self.action = action
        self.cost = cost

class Graph:
    def __init__(self, problem: SearchProblem, fringe):
        """
        Initializes the graph search with the specified problem and datastructure.

        problem: The problem in which to search
        fringe: The datastructure to use in seearch algorithm. This can be a stack, queue or priority queue.
        """
        self.problem = problem
        self.visited = set()
        self.fringe = fringe

    def search(self):
        # Start expansion with the start state and zero cost
        self.current = Edge(None, self.problem.getStartState(), None, 0)

        while not self.problem.isGoalState(self.current.target):
            # Expand the current node if we haven't visited it yet
            if self.current.target not in self.visited:
                self.expand(self.current)

            # Continue with the next node as determined by the datastructure we're using
            self.current = self.fringe.pop()

        # We've found the goal state, return the path
        return self.getPath(self.current)

    def expand(self, edge: Edge):
        # Keep track of the nodes we've visited
        self.visited.add(edge.target)

        for successor in self.problem.getSuccessors(edge.target):
            # Check if we've already visited this node
            if successor[0] not in self.visited:
                # If not, add the successor with necessary information to the stack/queue/priority queue
                self.fringe.push(Edge(edge, successor[0], successor[1], successor[2] + edge.cost))

    def getPath(self, target):
        path = list()
        # Track back from the target state, we've saved the edges that led there with the according action
        while target.action is not None:
            path.insert(0, target.action)
            target = target.source

        return path

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.
    """

    # Search with a stack (FIFO)
    graph = Graph(problem, util.Stack())

    return graph.search()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""

    # Search with a queue (FILO)
    graph = Graph(problem, util.Queue())

    return graph.search()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""

    # Use a priority queue with the cost so far to determine the priority
    graph = Graph(problem, util.PriorityQueueWithFunction(lambda edge : edge.cost))

    return graph.search()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    # Combine the cost so far with the heuristic to determine the priority
    graph = Graph(problem, util.PriorityQueueWithFunction(lambda edge : edge.cost + heuristic(edge.target, problem)))

    return graph.search()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
