# author = Thomas Haydn
# title  = Electricity market players behavior simulated with Q-learning
# This script simulates price bidding behavior of different players in a electricity market with uniform bidding.
# Each player is owner of multiple generators and can decide on a bidding strategy for each generator individually.
# Players use a simple reinforcement learning strategy (Q-learning) to improve from auction-round to auction-round.

# experiment => sum of all runs for certain parameter settings
# run => sum of all iterations
# iteration => One bidding round for a certain demand

import numpy as np
import random
import math
import itertools
import csv
from matplotlib import pyplot as plt
import time
import cProfile

"""Console settings--------------------------------------------------------------------------------------------------"""

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=10000000)
np.set_printoptions(linewidth=1000)

"""Classes-----------------------------------------------------------------------------------------------------------"""


class Player:

    def __init__(self, name, profit, generator_list, q_table, action, action_size,
                 gen_nmb, strategy_table, best_actions, best_action_list,
                 action_list):
        self.name = name
        self.profit = profit
        self.generator_list = generator_list
        self.q_table = q_table
        self.action = action
        self.action_size = action_size
        self.gen_nmb = gen_nmb
        self.strategy_table = strategy_table
        self.action_list = action_list
        self.best_actions = best_actions
        self.best_action_list = best_action_list

    def __repr__(self):
        return '{!r}'.format(self.name)


class Generator:

    def __init__(self, name, capacities, costs, current_price,
                 current_sup, sold_quantity, profit_gen, total_profit_gen, strategy, owner, order, price_list):
        self.name = name
        self.capacities = capacities
        self.costs = costs
        self.current_price = current_price
        self.current_sup = current_sup
        self.sold_quantity = sold_quantity
        self.profit_gen = profit_gen
        self.total_profit_gen = total_profit_gen
        self.strategy = strategy
        self.owner = owner
        self.order = order
        self.price_list = price_list

    def __repr__(self):
        return '{!r}'.format(self.name)


"""Initializing------------------------------------------------------------------------------------------------------"""


def initialize():
    # Sets up all things that have to be executed once
    # This happens at the start of each experiment

    check_costs_capacity()

    check_max_demand()

    set_generator_order()

    set_up_strategy_tables()

    set_up_best_action_list()

    set_up_generator_parameters()

    calculate_possible_prices()

    for gen in gen_list:
        print(gen.price_list)

def set_up():
    # Sets up player and generator parameters
    # This happens at the start of each run

    set_up_player_parameter()

    calculate_state_step_size()

def auction_round(index):
    # Represents one auction round with a certain demand (current_demand)
    # index represents the number of the auction_round within a run
    # This happens every iteration

    # print("Runde " + str(index))

    current_demand = demand_list[index]

    state = check_state(current_demand)

    pick_action(state, index)

    if index >= 0.98 * len(demand_list):
        for pla in player_list:
            pla.action_list.append(pla.action)

    set_generator_strategy()

    set_generator_price()

    market_clearing_with_tiebreaker(current_demand)

    update_q_values(state, index)

    gen_list.sort(key=lambda var: int(var.order))


"""Functions---------------------------------------------------------------------------------------------------------"""

def create_list(list_name, csv_sheet):
    # creates lists from csv files

    list_values = open(csv_sheet)
    for list_value in list_values.readlines():
        list_name.append(int(list_value))
    list_values.close


def set_up_generator_list():
    # creates a generator-object for each element in capacity_list

    generator_namelist = []
    for index, element in enumerate(capacity_list):
        generator_namelist.append("Generator" + str(index + 1))

    # creates a generator-object for each element in generator_namelist and adds it to a dct:
    generator_dct = {name: Generator(name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []) for name in generator_namelist}

    # Transforms dct into list:
    g_list = []
    for gen in generator_dct:
        g_list.append(generator_dct.get(gen))
    return g_list


def set_up_player_list():
    # creates player-objects

    player_namelist = []
    for index in range(PLAYER_NMB):
        player_namelist.append("Player" + str(index + 1))

    # creates a generator-object for each element in generator_namelist and adds it to a dct:
    player_dct = {name: Player(name, 0, [], 0, 0, 0, 0, 0, 0, 0, 0) for name in player_namelist}

    # Transforms dct into list:
    p_list = []
    for pla in player_dct:
        p_list.append(player_dct.get(pla))

    # Generates a generator list for each player:
    x = 0
    for index, nmb in enumerate(PLAYER_GEN_NMB):
        for counter in range(nmb):
            p_list[index].generator_list.append(gen_list[x])
            x += 1

    return p_list


def find_max_index(array):
    # finds indices of all max_values of an array. returns a list
    # row of array needs to be included (E.g. array[1, :])

    max_list = []
    max_val = np.max(array)  # finds only one maximum

    for index, element in enumerate(array):
        if element == max_val:
            max_list.append(index)
    return max_list


def check_costs_capacity():
    # checks if costs-list and capacities-list are the same length, since each generator needs exactly one value each

    if len(capacity_list) != len(costs_list):
        print("Capacity and costs can not be allocated")
    else:
        print("There are " + str(len(capacity_list)) + " generators")


def set_generator_order():
    # gives each generator an individual marker to reset the order later
    for index, gen in enumerate(gen_list):
        gen.order = index


def check_max_demand():
    # checks if generators can satisfy demand

    total_capacity = 0
    for capacity in capacity_list:
        total_capacity += capacity

    if max(demand_list) > total_capacity:
        print("Not enough generators!!!")


def set_up_best_action_list():
    # creates an array for each player
    # a row for each state, a column for each run
    # the array is filled with the action with the highest Q-value at the end of a single run
    for pla in player_list:
        pla.best_action_list = np.empty((STATE_SIZE, RUNS))


def set_up_generator_parameters():

    for index, gen in enumerate(gen_list):
        gen.capacities = capacity_list[index]
        gen.costs = costs_list[index]
        gen.current_sup = int(gen.capacities)


def set_up_player_parameter():

    for pla in player_list:
        pla.action_size = ACTION_PER_GEN ** pla.gen_nmb  # nmb of possible player_actions
        pla.q_table = np.zeros((STATE_SIZE, pla.action_size))  # creates individual Q_table
        # lists and variables for output
        pla.total_profit = 0
        pla.profit_list = []
        pla.profit_list_single = []
        pla.action_list = []


def set_up_strategy_tables():
    # strategy tables are used to convert a single player-action into individual generator actions
    # strategy tables represent all possible generator-action combinations for one player

    for pla in player_list:
        pla.gen_nmb = len(pla.generator_list)
        pla.strategy_table = create_strategy_table(pla.gen_nmb)


def create_strategy_table(gen_nmb):
    # uses itertools to create strategy tables

    action_list = [index for index in range(ACTION_PER_GEN)]
    temp = (itertools.product(action_list, repeat=gen_nmb))
    return np.array(list(temp)).T


def calculate_state_step_size():
    # defines the demand range of a state

    demand_range = max(demand_list) - MIN_DEMAND
    state_step_size = demand_range / STATE_SIZE
    return state_step_size


def check_state(current_demand):
    # returns the current state

    if current_demand == MIN_DEMAND:  # else statement does not work for current_demand == min(demand_list)
        state = 0
    else:
        state_step = calculate_state_step_size()
        state = math.ceil((current_demand - MIN_DEMAND) / state_step) - 1
    return state


def update_epsilon_value(index):
    # updates the epsilon value based on index within one run

    epsilon = EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN) * np.exp(- EPSILON_DECAY_RATE * index)
    return epsilon


def pick_action(state, index):
    # Decides for exploitation or exploration
    # Exploitation picks the action with the highest Q-value. If there are two or more equivalent actions one is
    # chosen randomly. Exploration chooses a random action.

    epsilon = update_epsilon_value(index)

    # print('epsilon: ' + str(epsilon))
    for pla in player_list:

        exp_exp_trade_off = random.uniform(0, 1)  # random variable to decide between exploitation and exploration

        if exp_exp_trade_off > epsilon:  # Exploitation:
            best_actions = find_max_index(pla.q_table[state, :])  # Creates a list with index of max_Q_values
            pla.action = random.choice(best_actions)
            # print(str(pla.name) + ': Exploit')
        else:  # Exploration:
            pla.action = random.randint(0, pla.action_size - 1)
            # print(str(pla.name) + ': Random')


def set_generator_strategy():
    # translates player action into generator strategy

    for pla in player_list:
        for index, gen in enumerate(pla.generator_list):
            gen.strategy = pla.strategy_table[index, pla.action]


def set_generator_price():
    # translates a generator strategy in the suitable price from the price_list

    for gen in gen_list:
        gen.current_price = gen.price_list[gen.strategy]


def calculate_possible_prices():

    for gen in gen_list:
        price_step_range = (highest_costs * PRICE_COEF - gen.costs) / (ACTION_PER_GEN - 1)

        for index in range(ACTION_PER_GEN):
            price_value = gen.costs + index * price_step_range
            gen.price_list.append(price_value)


def market_clearing_with_tiebreaker(current_demand):
    # Executes the market clearing process. If there are two or more generators who sell at selling price, the missing
    # demand is split among them equally.

    gen_list.sort(key=lambda var: var.current_price)  # sort gen_list from lowest price to highest

    sum_sup = 0  # sum of total supplies already used
    selling_price = 0  # price of the most expensive selling generator

    for gen in gen_list:
        if sum_sup < current_demand:  # finding selling price for next step
            selling_price = gen.current_price
            sum_sup += gen.current_sup  # Updating sum of total supply:

    selling_price_list.append(selling_price)  # for output

    sum_sup = 0  # reset for real clearing process
    tie_capacity_sum = 0  # number of generators selling at selling price

    for gen in gen_list:

        if gen.current_price < selling_price:
            gen.sold_quantity = gen.current_sup  # all generators under selling price sell their whole capacity
            sum_sup += gen.current_sup  # sum_sup update

        if gen.current_price == selling_price:
            tie_capacity_sum += gen.capacities

    needed_tie_capacity = current_demand - sum_sup
    needed_percentage = needed_tie_capacity / tie_capacity_sum

    for gen in gen_list:
        if gen.current_price == selling_price:
            gen.sold_quantity = gen.capacities * needed_percentage

    for gen in gen_list:
        gen.profit = gen.sold_quantity * (selling_price - gen.costs)  # generator profit calculation
        gen.sold_quantity = 0  # reset for next round

    # for gen in gen_list:
        # print(str(gen.name) + ": action: " + str(gen.strategy) + ", price: " + str(gen.current_price) + ", profit: " + str(gen.profit))

    for pla in player_list:
        pla.profit = 0
        for gen in pla.generator_list:
            pla.profit += gen.profit  # player profit calculation

        pla.total_profit += pla.profit  # player total profit calculation


def update_q_values(state, index):

    for pla in player_list:
        # More or less Bellman equation:
        pla.q_table[state, pla.action] = pla.q_table[state, pla.action] + \
                                                  LEARNING_RATE * (pla.profit - pla.q_table[state, pla.action])


def update_best_action_list(run_index):
    # fills best action list with winning action for each player in each state at the end of a run
    # if there are two or more equivalent actions at the end of a run, one is chosen randomly

    for pla in player_list:
        for state in range(STATE_SIZE):
            best_actions = find_max_index(pla.q_table[state, :])
            chosen_best_action = random.choice(best_actions)
            pla.best_action_list[state, run_index] = chosen_best_action


def translate_strategy_combination(play, strategy):
    choosen_strategy = play.strategy_table[:, int(strategy)]
    for index, gen in enumerate(play.generator_list):
        price = gen.price_list[int(choosen_strategy[index])]
        print("Generator" + str(gen.order + 1) + " => action " + str(choosen_strategy[index])
              + " => price: " + (str(price)))


def find_dominant_strategy(listx):

    count_list = []
    for element in range(len(listx)):
        count_list.append(listx.count(listx[element]))

    maximum = (max(count_list))

    for element in range(len(count_list)):
        if listx.count(listx[element]) == maximum:
            dominant_strategy_combination = listx[element]
            print('dominant strategy is' + str(listx[element]) + 'with: ' + str(maximum) + ' appearances')
            break
    s = maximum / (0.02 * len(demand_list))
    print('s = ' + str(s))
    return dominant_strategy_combination, s


def count_combinations(listx):

    modified_list = []
    for element in listx:
        modified_list.append(tuple(element))

    modified_list = sorted(modified_list)

    listx = []
    for element in modified_list:
        listx.append(list(element))

    count_list = []

    for element in range(len(listx)):
        count_list.append(listx.count(listx[element]))

    double_check_list = []

    for index, value in enumerate(count_list):
        if double_check_list.count(listx[index]) < 1:
            print('Combination ' + str(listx[index]) + ' was played ' + str(count_list[index]) + ' times')
            double_check_list.append(listx[index])


"""Global Variables--------------------------------------------------------------------------------------------------"""
# Collection of global variables and lists. Parameter settings can be changed here.

ACTION_PER_GEN = 4
PRICE_COEF = 2
STATE_SIZE = 1
PLAYER_NMB = 2
PLAYER_GEN_NMB = [2,2]

RUNS = 50

# Exploration, Exploitation trade of
EPSILON_MAX = 1
EPSILON_MIN = 0.00
EPSILON_DECAY_RATE = 5 / 150000
LEARNING_RATE = 0.00475 # Defines the weight of new experiences.

plt.style.use('seaborn-whitegrid')

demand_list = []
create_list(demand_list, "demand_45000.csv")

capacity_list = []
create_list(capacity_list, "capacity.csv")

costs_list = []
create_list(costs_list, "costs.csv")

highest_costs = max(costs_list)

gen_list = set_up_generator_list()

player_list = set_up_player_list()

selling_price_list = []

tick_list = [i for i, nmb in enumerate(demand_list)]

MIN_DEMAND = min(demand_list)

action_combination_list = []

dominant_action_list = []
stability_parameter_list = []

"""Execution---------------------------------------------------------------------------------------------------------"""

# helping tools:
tic = time.time()
pr = cProfile.Profile()
pr.enable()

initialize()  # happens at the start of the experiment

for i in range(RUNS):  # happens at the start of each run

    set_up()

    selling_price_list = []  # has to be empty at the start of the run

    action_combination_list = []

    for j, demand in enumerate(demand_list):

        # print('iteration:' + str(j))

        combination_list = []  # list of winning strategy combinations, has to be empty at the start of each iteration

        auction_round(j)  # happens each iteration

        # for pla in player_list:
        #     print(str(pla.name) + ': ' + str(pla.q_table))
        # print('-----------------------------------------------------------------------------')
        if j >= 0.98 * len(demand_list):
            for pla in player_list:
                combination_list.append(pla.action_list[math.floor(j - 0.98 * len(demand_list))])

            action_combination_list.append(combination_list)


    action_set_list = []

    update_best_action_list(i)

    dom_strat, s = find_dominant_strategy(action_combination_list)
    dominant_action_list.append(dom_strat)
    stability_parameter_list.append(s)
    print(action_combination_list)


    i += 1

    print(i)

# for player in player_list:
#     print(str(player.name) + ": " + str(player.best_action_list))
#     print(str(player.name) + ": " + str(player.q_table))

# for pla in player_list:
#     print(pla.q_value_matrix)

   # print(str(translate_strategy_combination(player, player.best_action_list[-1])))
    # for element in player.best_action_list[0]:
    #     print(translate_strategy_combination(player, element))

print(dominant_action_list)
count_combinations(dominant_action_list)
print(stability_parameter_list)

# pr.disable()
# pr.print_stats()

toc = time.time()
print("time: " + str(toc - tic) + " seconds")

