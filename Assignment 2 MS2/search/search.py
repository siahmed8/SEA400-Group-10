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
from typing import List

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




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    stack = util.Stack() #LIFO C, D -> C,F -> C,G (end) -> G,H
    visited = set()

    initial_state = problem.getStartState()
    stack.push((initial_state, []))

    while not stack.isEmpty():
        state, path = stack.pop()

        if state in visited:
            continue

        visited.add(state)

        if problem.isGoalState(state):
            return path

        for successor, action, stepCost in problem.getSuccessors(state):
            if successor not in visited:
                stack.push((successor, path + [action]))

    return [] # if no solution found

    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    queue = util.Queue() #FIFO
    visited = set()

    initial_state = problem.getStartState()
    queue.push((initial_state, []))

    while not queue.isEmpty():
        state, path = queue.pop()

        if state in visited:
            continue

        visited.add(state)

        if problem.isGoalState(state):
            return path

        for successor, action, stepCost in problem.getSuccessors(state):
            if successor not in visited:
                queue.push((successor, path + [action]))

    return [] # if no solution found

    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    pq = util.PriorityQueue()
    visited = set()
    initial_state = problem.getStartState()

    pq.push((initial_state, [], 0), 0)

    while not pq.isEmpty():
        state, path, cost = pq.pop()

        if state in visited:
            continue

        visited.add(state)

        if problem.isGoalState(state):
            return path

        for successor, action, stepCost in problem.getSuccessors(state):
            if successor not in visited:
                new_cost = cost + stepCost
                pq.push((successor, path + [action], new_cost), new_cost)

    return []
    util.raiseNotDefined()

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""

    # priority queue ensures we always expand the node with the lowest total cost so far
    priority_queue = util.PriorityQueue()
    # dict to keep track of the best cost to reach each state found so far
    best_cost = {}

    initial_state = problem.getStartState()  # get starting string of the problem

    """
    Push the starting state into the priority queue.
    stored as: (state, path_to_state, cost_so_far)
    priority = cost_so_far + heuristic estimate
    """
    priority_queue.push(
        (initial_state, [], 0),
        heuristic(initial_state, problem)
    )
    best_cost[initial_state] = 0  # cost to reach initial state is zero

    # continue searching while other states available to explore
    while not priority_queue.isEmpty():
        # pop state with lowest cost+heuristic
        state, path, cost_so_far = priority_queue.pop()

        # skip state if we have already found a cheaper path to it
        if cost_so_far > best_cost.get(state, float('inf')):
            continue

        # if goal reached, return the solution path
        if problem.isGoalState(state):
            return path

        # expand current state to explore successors
        for successor, action, stepCost in problem.getSuccessors(state):

            # calculate new path cost for successor
            new_cost = cost_so_far + stepCost

            # only consider successor if this path is better than any previously found
            if successor not in best_cost or new_cost < best_cost[successor]:

                """
                Calculate priority using A* formula.
                f(n) = g(n) + h(n)
                """
                # push successor into priority queue with computed priority
                priority = new_cost + heuristic(successor, problem)
                priority_queue.push(
                    (successor, path + [action], new_cost),
                    priority
                )
                best_cost[successor] = new_cost

    return []  # return empty path for no solutions

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch