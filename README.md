# Q-learning-electricity-market
# author = Thomas Haydn

# This algorithm simulates price bidding behavior of different players in a electricity market with uniform bidding.
# Each player is owner of multiple generators and can decide on a bidding strategy for each generator individually.
# Players use a simple reinforcement learning strategy (Q-learning) to improve from auction-round to auction-round.

# How to use it: 
# Parameter settings can be changed in order to apply the algorithm to a specific problem. Parameters can be changed 
# in the sector 'Global Variables' (line 457). Moreover 3 csv. files are needed for data input. 

# Changeable parameters: 

# ACTION_PER_GEN...Number of allowed price bids per generator. (example = 4)
# PRICE_COEF...Multiplier for highest allowed price bid. (example = 2)
# STATE_SIZE...Number of states. For a constant demand a STATE_SIZE of 1 is reasonable. For altering demands more states 
# improve agents' performance but more iterations are needed. (example = 1)
# PLAYER_NMB...Number of simulated players. Games wih more than 3 players tend to become intraceable. (example = 2) 
# PLAYER_GEN_NMB...Has the form of a list and summarizes the number of generators each player owns. 
# (example: 2 players, player1 owns 2 generators, player2 owns 3 generators = [2,3])

# RUNS... Number of experiment runs. Each experiment should be run multiple times. (example = 50)

# EPSILON_MAX... Maximal value epsilon can reach. A value of 1 is highly recommended.
# EPSILON_MIN...Minimal value epsilon can reach. A value if 0 is highly recommended 
# EPSILON_DECAY_RATE...Regulates epsilon decay process. We recommend to select a EPSILON_DECAY_RATE according to: 
# EPSILON_DECAY_RATE = 5/iterations
# LEARNING_RATE...Fegulates how fast new experiences influence according Q-values. For bigger set-ups a lower 
# lower learning rate is sufficient. Learning rates should not be set higher than 0.1. 

# Needed csv files: 

# demand_45000.csv...Has to include one demand value for each iteration. Each iterations is represented by one row. 

# capacity.csv...Has to include one capacity value for each generator. Each generator is represented by one row. 
# The first row is distributed to the first player's generator and so on. 

# costs.csv...Has to include one marginal cost value for one produced unit of electricity for each generator. 
# Each generator is represented by one row. The first row is distributed to the first player's generator and so on. 