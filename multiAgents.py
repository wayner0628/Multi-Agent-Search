from util import manhattanDistance
from game import Directions
import random
import util
from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min(
            [manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food)
                                  for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(
            newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(
            newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        """---------------Explanation----------------
        The function value() would determine whether it's a min-layer, max-layer,
        or terminal; agent = 0 represents pacman, which is a max-player, so call the
        function max_value(), on the contrary, the other agents are ghosts, calling the
        function min_value(), and if the current depth reach the limit or the game is
        over, return the action.
        The function max_value() would initialize the action as a list of an empty 
        direction and a negative infinite state value. Then, visit all the possible 
        actions, checking the substates' values by calling the value() again, since agents
        with index > 0 are ghosts, value() would call min_value; and the datatype of 
        subvalue is list representing the return value is a combination of direction
        and value, otherwise, it is an evaluation of state, pick the value in both cases,
        if the value is greater than a current one, replace it.
        The funtion min_value() does the same work as max_value(), however, it calls 
        value() with agent+1, in normal case, it receives an action returned by another
        ghost, since there may be many ghosts, one picks an action followed by another,
        and only if the agent index passed is greater than the actual number, representing
        all the agents are done, so the depth would go down one, and reset index to 0, 
        start from pacman again.
        """
        # Begin your code (Part 1)
        def value(gameState, deep, agent):
            if agent >= gameState.getNumAgents():
                agent = 0
                deep += 1
            if (deep == self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            elif (agent == 0):
                return max_value(gameState, deep, agent)
            else:
                return min_value(gameState, deep, agent)

        def max_value(gameState, deep, agent):
            dir_val = ['', -float("inf")]
            pacActions = gameState.getLegalActions(agent)
            if not pacActions:
                return self.evaluationFunction(gameState)
            for action in pacActions:
                subState = gameState.getNextState(agent, action)
                subValue = value(subState, deep, agent+1)
                if type(subValue) is list:
                    tmp = subValue[1]
                else:
                    tmp = subValue
                if tmp > dir_val[1]:
                    dir_val = [action, tmp]
            return dir_val

        def min_value(gameState, deep, agent):
            dir_val = ['', float("inf")]
            ghostActions = gameState.getLegalActions(agent)
            if not ghostActions:
                return self.evaluationFunction(gameState)
            for action in ghostActions:
                subState = gameState.getNextState(agent, action)
                subValue = value(subState, deep, agent+1)
                if type(subValue) is list:
                    tmp = subValue[1]
                else:
                    tmp = subValue
                if tmp < dir_val[1]:
                    dir_val = [action, tmp]
            return dir_val

        return value(gameState, 0, 0)[0]
        # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """-----------------Explanation------------------
        Things here are much similar to those at MinimaxAgent, it simply arranges
        part of min_value() and max_value() as using alpha and beta to prune.
        In max_value(), if the return value from substate is greater than beta, 
        directly return; if it's greater than alpha, update alpha's value.
        In min_value(), if the return value from substate is less than alpha, 
        directly return; if it's less than beta, update beta's value.
        """
        # Begin your code (Part 2)
        def value(gameState, deep, agent, alpha, beta):
            if agent >= gameState.getNumAgents():
                agent = 0
                deep += 1
            if (deep == self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            elif (agent == 0):
                return max_value(gameState, deep, agent, alpha, beta)
            else:
                return min_value(gameState, deep, agent, alpha, beta)

        def max_value(gameState, deep, agent, alpha, beta):
            dir_val = ['', -float("inf")]
            pacActions = gameState.getLegalActions(agent)
            if not pacActions:
                return self.evaluationFunction(gameState)
            for action in pacActions:
                subState = gameState.getNextState(agent, action)
                subValue = value(subState, deep, agent+1, alpha, beta)
                if type(subValue) is list:
                    tmp = subValue[1]
                else:
                    tmp = subValue
                if tmp > dir_val[1]:
                    dir_val = [action, tmp]
                if tmp > beta:
                    return dir_val
                alpha = max(alpha, tmp)
            return dir_val

        def min_value(gameState, deep, agent, alpha, beta):
            dir_val = ['', float("inf")]
            ghostActions = gameState.getLegalActions(agent)
            if not ghostActions:
                return self.evaluationFunction(gameState)
            for action in ghostActions:
                subState = gameState.getNextState(agent, action)
                subValue = value(subState, deep, agent+1, alpha, beta)
                if type(subValue) is list:
                    tmp = subValue[1]
                else:
                    tmp = subValue
                if tmp < dir_val[1]:
                    dir_val = [action, tmp]
                if tmp < alpha:
                    return dir_val
                beta = min(beta, tmp)
            return dir_val

        return value(gameState, 0, 0, -float("inf"), float("inf"))[0]
        # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """-----------------Explanation---------------
        Things here are much similar to those at MinimaxAgent, it simply arranges
        part of min_value() as ghosts are no longer min_players, but act randomly.
        In exp_value(), all the possible actions have a chance of 1/number of actions 
        to occur, so any direction can be chosen as it is just a guess, which really 
        counts is the mean value of this agent's possible actions, the mean is returned.
        """
        # Begin your code (Part 3)
        def value(gameState, deep, agent):
            if agent >= gameState.getNumAgents():
                agent = 0
                deep += 1
            if (deep == self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            elif (agent == 0):
                return max_value(gameState, deep, agent)
            else:
                return exp_value(gameState, deep, agent)

        def max_value(gameState, deep, agent):
            dir_val = ['', -float("inf")]
            pacActions = gameState.getLegalActions(agent)
            if not pacActions:
                return self.evaluationFunction(gameState)
            for action in pacActions:
                subState = gameState.getNextState(agent, action)
                subValue = value(subState, deep, agent+1)
                if type(subValue) is list:
                    tmp = subValue[1]
                else:
                    tmp = subValue
                if tmp > dir_val[1]:
                    dir_val = [action, tmp]
            return dir_val

        def exp_value(gameState, deep, agent):
            dir_val = ['', 0]
            ghostActions = gameState.getLegalActions(agent)
            if not ghostActions:
                return self.evaluationFunction(gameState)
            p = 1.0/len(ghostActions)
            for action in ghostActions:
                subState = gameState.getNextState(agent, action)
                subValue = value(subState, deep, agent+1)
                if type(subValue) is list:
                    val = subValue[1]
                else:
                    val = subValue
                dir_val[0] = action
                dir_val[1] += val * p
            return dir_val

        return value(gameState, 0, 0)[0]
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    """-----------------Explanation-----------------
    Collect some information from the built-in function:
    food's position -> food's distance from pacman
    capsules' position -> capsules' distance from pacman
    ghosts' position -> ghosts' distance from pacman
    timer: the remain scaredTime
    Then the first idea of evaluation is when the least timer is greater than the least
    distance of ghost, there is a chance to capture the ghost with 200 scores, and the 
    remain distance to catch the ghost is also taken into account. Using min(timer) is 
    because the ghost will revive after being eaten by pacman, and its scaredtime is
    reset, so the scaredtimes are not concurrent, and tracing every ghost's scaredtime
    is annoyed, so using the least one is a much easy but safty approach. On the other
    hand, since the ghosts move randomly, they get closer or further to pacman with the 
    same chance, so I suppose that it remain the same, the timer is compared directly
    with the distance.
    The second idea reacts under timer = 0, if the distance is 0, pacman is catched by
    the ghost, so return negative infinity, and if distance is less than 5, and the
    capsules is closer than the ghost(if no capsule remain, its distance is infinity)
    , then it's a chance for pacman to eat capsules and hunt the ghost, so it's also
    rewarded by 100(state value), the distance to get capsules is considered. Finally,
    if the distance is 1, it is considered to be dangerous and not recommended. In the
    remain situations, only the food distance is considered, the pacman would tend to 
    get to a closer food's position. 
    """
    # Begin your code (Part 4)
    foodPos = currentGameState.getFood().asList()
    foodDist = []
    ghostPos = currentGameState.getGhostPositions()
    capPos = currentGameState.getCapsules()
    currentPos = list(currentGameState.getPacmanPosition())
    ghostDist = []
    capDist = []
    timer = []
    num = currentGameState.getNumAgents()
    for i in range(1, num):
        timer.append(currentGameState.getGhostState(i).scaredTimer)
    for food in foodPos:
        foodDist.append(manhattanDistance(food, currentPos))
    if not foodDist:
        foodDist.append(0)
    for cap in capPos:
        capDist.append(manhattanDistance(cap, currentPos))
    if not capDist:
        capDist.append(float("inf"))
    for ghost in ghostPos:
        ghostDist.append(manhattanDistance(ghost, currentPos))
    if(min(timer) >= min(ghostDist)):
        return currentGameState.getScore() - min(ghostDist) + 200
    if(min(timer) == 0):
        if(min(ghostDist) == 0):
            return -float("inf")
        elif(min(ghostDist) < 5 and min(ghostDist) > min(capDist)):
            return currentGameState.getScore() - min(capDist) + 100
        elif(min(ghostDist) == 1):
            return currentGameState.getScore() - 500

    return currentGameState.getScore() - min(foodDist) + 10
    # End your code (Part 4)


# Abbreviation
better = betterEvaluationFunction
