# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util
import gridworld
from learningAgents import ValueEstimationAgent
import collections

# VOOR DEBUGGEN
mdpFunction = gridworld.getBridgeGrid
mdp = mdpFunction()
mdp.setLivingReward(0)
mdp.setNoise(0.2)
env = gridworld.GridworldEnvironment(mdp)
#a = ValueIterationAgent(mdp, 0.9, 100)

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()
        self.tempvalues = util.Counter()
        self.qvalues = util.Counter()

        # Run the value iterations
        for i in range(iterations):
            self.runValueIteration()
   

    def runValueIteration(self):
        for state in self.mdp.getStates():
            bestvalue = float('-inf')
            #V(s) of a state can only change if an action from s can lead to a state s' with V(s) != 0
            #So if the values of successor states are 0, than the sum will be 0. Then we don't count this action
            check = 0

            for action in self.mdp.getPossibleActions(state):
                transitions = self.mdp.getTransitionStatesAndProbs(state, action)

                #If check is ever not 0, then there is a possible successor state s' with V(s') <= 0. So we count the action.
                if check == 0:
                    check = (sum((self.getValue(transition[0]) != 0) for transition in transitions) != 0) # (1 if true)

                #If the state is terminal (so strictly speaking, if it has possible action 'exit', V = reward
                if len(transitions) == 1 and self.mdp.isTerminal(transitions[0][0]):
                    bestvalue = self.mdp.getReward(state, action, 'TERMINAL_STATE')
                    check = 1
                else:
                    value = sum((transition[1] *
                                (self.mdp.getReward(state, action, transition[0]) + self.discount * self.getValue(transition[0])))
                                for transition in transitions)
                    if(value > bestvalue):
                        bestvalue = value

                if check == 1:
                    self.tempvalues[state] = bestvalue

        # Only update the values after the iteration, otherwise we will use intermediate values to update
        for state in self.mdp.getStates():
            self.values[state] = self.tempvalues[state]


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)

        values = [transition[1] * (self.mdp.getReward(state, action, transition[0]) + self.discount * self.getValue(transition[0])) for transition in transitions]

        return sum(values)


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actions = self.mdp.getPossibleActions(state)

        # This is the only legal action
        if 'exit' in actions:
            return 'exit'

        # Default is action None, initialize the expected value at minus infinity so negative values are still counted
        qval = -float('inf')
        best = None

        for action in actions:
            qvalnew = self.computeQValueFromValues(state, action)
            # Select the new action if it's better than any of the previous ones
            if qvalnew > qval:
                qval = qvalnew
                best = action

        return best

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

a = ValueIterationAgent(mdp, 0.9, 100)
