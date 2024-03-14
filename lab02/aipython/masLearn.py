# masLearn.py - Multiagent learning
# AIFCA Python code Version 0.9.12 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from display import Displayable
import matplotlib.pyplot as plt
from rlProblem import RL_agent

class SimulateGame(Displayable):
    def __init__(self, game, agent_types):
        #self.max_display_level = 3
        self.game = game
        self.agents = [agent_types[i](game.players[i], game.actions[i], 0) for i in range(game.num_agents)] # list of agents
        self.action_dists = [{act:0 for act in game.actions[i]} for i in range(game.num_agents)]
        self.action_history = []
        self.state_history = []
        self.reward_history = []
        self.dist = {}
        self.dist_history = []
        self.actions = tuple(ag.initial_action(game.initial_state) for ag in self.agents)
        self.num_steps = 0

    def go(self, steps):
        for i in range(steps):
            self.num_steps += 1
            (self.rewards, state) = self.game.play(self.actions)
            self.display(3, f"In go rewards={self.rewards}, state={state}")
            self.reward_history.append(self.rewards)
            self.state_history.append(state)
            self.actions = tuple(agent.select_action(reward, state)
                                     for (agent,reward) in zip(self.agents,self.rewards))
            self.action_history.append(self.actions)
            for i in range(self.game.num_agents):
                 self.action_dists[i][self.actions[i]] += 1
            self.dist_history.append([{a:i for (a,i) in elt.items()} for elt in self.action_dists]) # deep copy
        #print("Scores:", ' '.join(f"{self.agents[i].role} average reward={ag.total_score/self.num_steps}" for ag in self.agents))
        print("Distributions:", ' '.join(str({a:self.dist_history[-1][i][a]/sum(self.dist_history[-1][i].values()) for a in self.game.actions[i]})
                                             for  i in range(self.game.num_agents)))
        #return self.reward_history, self.action_history

    def action_dist(self,which_actions=[1,1]): 
        """ which actions is  [a0,a1]
        returns the empirical distribution of actions for agents,
           where ai specifies the index of the actions for agent i
        remove this???
        """
        return [sum(1 for a in sim.action_history
                        if a[i]==gm.actions[i][which_actions[i]])/len(sim.action_history)
                    for i in range(2)]

    def plot_dynamics(self, x_action=0, y_action=0):
        plt.ion()  # make it interactive
        agents = self.agents
        x_act = self.game.actions[0][x_action]
        y_act = self.game.actions[1][y_action]
        plt.xlabel(f"Probability {self.game.players[0]} {self.agents[0].actions[x_action]}")
        plt.ylabel(f"Probability {self.game.players[1]} {self.agents[1].actions[y_action]}")
        plt.plot([self.dist_history[i][0][x_act]/sum(self.dist_history[i][0].values()) for i in range(len(self.dist_history))],
                 [self.dist_history[i][1][y_act]/sum(self.dist_history[i][1].values()) for i in range(len(self.dist_history))])
        #plt.legend()
        #plt.savefig('soccerplot.pdf') # if you want to save plot
        plt.show()

class ShoppingGame(Displayable):
    def __init__(self):
        self.num_agents = 2
        self.states = ['s']
        self.initial_state = 's'
        self.actions = [['shopping', 'football']]*2
        self.players = ['football-preferrer goes to', 'shopping-preferrer goes to']

    def play(self, actions):
        """Given (action1,action2) returns (resulting_state, (reward1, reward2))
        """
        return ({('football', 'football'): (2, 1),
                 ('football', 'shopping'): (0, 0),
                 ('shopping', 'football'): (0, 0),
                 ('shopping', 'shopping'): (1, 2)
                     }[actions], 's')

class SoccerGame(Displayable):
    def __init__(self):
        self.num_agents = 2
        self.states = ['s']
        self.initial_state = 's'
        self.initial_state = 's'
        self.actions = [['right', 'left']]*2
        self.players = ['goalkeeper', 'kicker']

    def play(self, actions):
        """Given (action1,action2) returns (resulting_state, (reward1, reward2))
        resulting state is 's'
        """
        return ({('left', 'left'): (0.6, 0.4),
                 ('left', 'right'): (0.3, 0.7),
                 ('right', 'left'): (0.2, 0.8),
                 ('right', 'right'): (0.9,0.1)
               }[actions], 's')
               
class GameShow(Displayable):
    def __init__(self):
        self.num_agents = 2
        self.states = ['s']
        self.initial_state = 's'
        self.actions = [['takes', 'gives']]*2
        self.players = ['Agent 1', 'Agent 2']

    def play(self, actions):
        return ({('takes', 'takes'): (1, 1),
                ('takes', 'gives'): (11, 0),
                ('gives', 'takes'): (0, 11),
                ('gives', 'gives'): (10, 10)
               }[actions], 's')
               

class UniqueNEGameExample(Displayable):
    def __init__(self):
        self.num_agents = 2
        self.states = ['s']
        self.initial_state = 's'
        self.actions = [['a1', 'b1', 'c1'],['d2', 'e2', 'f2']]
        self.players = ['agent 1 does', 'agent 2 does']

    def play(self, actions):
        return ({('a1', 'd2'): (3, 5),
                 ('a1', 'e2'): (5, 1),
                 ('a1', 'f2'): (1, 2),
                 ('b1', 'd2'): (1, 1),
                 ('b1', 'e2'): (2, 9),
                 ('b1', 'f2'): (6, 4),
                 ('c1', 'd2'): (2, 6),
                 ('c1', 'e2'): (4, 7),
                 ('c1', 'f2'): (0, 8)
                     }[actions], 's')

# Choose a game:
# gm = ShoppingGame()
# gm = SoccerGame()
# gm = GameShow()
# gm = UniqueNEGameExample()

from rlQLearner import Q_learner
from rlProblem import RL_agent
from rlStochasticPolicy import StochasticPIAgent
# Choose one of the combinations of learners:
# sim=SimulateGame(gm,[StochasticPIAgent, StochasticPIAgent]); sim.go(10000)
# sim= SimulateGame(gm,[Q_learner, Q_learner]); sim.go(10000)
# sim=SimulateGame(gm,[Q_learner, StochasticPIAgent]); sim.go(10000)


# sim.plot_dynamics()

# empirical proportion that agents did their action at index 1:
# sim.action_dist([1,1])

# (unnormalized) empirical distribution for agent 0
# sim.agents[0].dist
