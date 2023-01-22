# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import util
import random

# QLearnAgent
#
class QLearnAgent(Agent):
    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        self.scoreTracker = 0
        self.stateTracker = 0
        self.lastaction = 0
        # Count the number of games we have played
        self.episodesSoFar = 0
        # Initialize the Qvalue array, the initial Qvalue array is an empty {}
        # And fill and update it in the following steps
        self.Qvalue = util.Counter()

    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1
    def getEpisodesSoFar(self):
        return self.episodesSoFar
    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value
    def getAlpha(self):
        return self.alpha
    def setAlpha(self, value):
        self.alpha = value
    def getGamma(self):
        return self.gamma
    def getMaxAttempts(self):
        return self.maxAttempts

    #Store the score
    def setScoreTracker(self, score):
        self.scoreTracker = score
    #Return the stored score
    def getScoreTracker(self):
        return self.scoreTracker

    #Store the state
    def setStateTracker(self, state):
        self.stateTracker = state
    #Return the stored state
    def getStateTracker(self):
        return self.stateTracker

    #Return the Qvalue
    def getQvalue(self,state,action):
        return self.Qvalue[(state,action)]

    #If the state is fixed, loop the all legal actions, return the MaxQvalue
    def getMaxQvalue(self, state):
        #Get all the legal actions
        legal = state.getLegalPacmanActions()
        # Remove the Stop
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        #The final state is that the program is finished,so it doesn't has the next action
        #self.getQvalue can't return Qvalue, it is [], so return 0
        if [self.getQvalue(state, action) for action in legal] == []:
            return  0
        # If it is not final state,just return the max Qvalue in the actions
        else:
            return max([self.getQvalue(state, action) for action in legal])

    #Return the best move
    def getBestmove(self,state):
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        #Set the initial bestmove is None
        Bestmove = None
        #Return the max Qvalue based on state
        MaxQvalue = self.getMaxQvalue(state)
        for action in legal:
            Qvalue = self.getQvalue(state, action)
            #Find which action's Qvalue is the max Qvalue, and return this action
            if Qvalue == MaxQvalue:
                Bestmove = action
            else:
                continue
        return Bestmove

    #According to the Q-learning algorithm, set the Q-learning function
    #Becaue I can't know the next state, so I use the current state to replace the next state
    #The last state and last action to replace the current state and current action.
    #Store the last state by the function:setStateTracker()
    #Return the last state by the function:getStateTracker()
    def update(self, laststate, lastaction, state, reward):
        #Get the last Qvalue based on the laststate and lastaction.
        Qvalue_last = self.getQvalue(laststate, lastaction)
        #Get the max Qvalue by the function getMaxQvalue based on the current state.
        Qvalue_current = self.getMaxQvalue(state)
        #The Update algorithm
        update = self.alpha * (reward + self.gamma * Qvalue_current - Qvalue_last)
        # Update the Qvalue
        self.Qvalue[(laststate, lastaction)] += update

    #Get the action
    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Get the last action
        self.lastaction = state.getPacmanState().configuration.direction
        # Get the current score
        score_current = state.getScore()
        # The current reward is that the current score minus the last score
        reward = score_current - self.getScoreTracker()
        # Store the current score for the next calculation
        self.setScoreTracker(score_current)
        # Update the Qvalue of last state and last action
        self.update(self.getStateTracker(), self.lastaction, state, reward)
        # Store the current state for the next calculation
        self.setStateTracker(state)

        #If it is exploration, just return the random choice
        if util.flipCoin(self.epsilon):
            print "Legal moves: ", legal
            print "Pacman position: ", state.getPacmanPosition()
            print "Ghost positions:", state.getGhostPositions()
            print "Food locations: "
            print state.getFood()
            print "Score: ", state.getScore()
            return random.choice(legal)
        #If it is exploitation, return the best move
        else:
            print "Legal moves: ", legal
            print "Pacman position: ", state.getPacmanPosition()
            print "Ghost positions:", state.getGhostPositions()
            print "Food locations: "
            print state.getFood()
            print "Score: ", state.getScore()
            return self.getBestmove(state)

        # Handle the end of episodes
        # This is called by the game after a win or a loss.
    def final(self, state):
        print "A game just ended!"
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        # Get the last action
        self.lastaction = state.getPacmanState().configuration.direction
        # Get the current score
        score_current = state.getScore()
        # The current reward is that the current score minus the last score
        reward = score_current - self.getScoreTracker()
        # Update the Qvalue of last state and last action
        self.update(self.getStateTracker(), self.lastaction, state, reward)
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg, '-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)
    # Handle the end of episodes
    # This is called by the game after a win or a loss.
