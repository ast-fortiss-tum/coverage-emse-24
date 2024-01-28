import matplotlib
matplotlib.use('TkAgg')

#from simulation.prescan_simulation import *

import sys
sys.path.insert(0, "problem/mnist/")

import pymoo


from model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from model_ga.result  import ResultExtended
pymoo.core.result.Result = ResultExtended

from model_ga.problem import ProblemExtended
pymoo.core.problem.Problem = ProblemExtended

from algorithm.ps_grid import PureSamplingGrid
from algorithm.ps_rand import PureSamplingRand

import argparse
import logging as log
import os
import re
import sys

from algorithm.nsga2_dt_sim import *
from algorithm.nsga2_sim import *

from default_experiments import *
from utils import log_utils

os.chmod(os.getcwd(), 0o777)

from pymoo.config import Config

log_utils.setup_logging(os.getcwd() + os.sep + "log.txt")

Config.warnings['not_compiled'] = False


results_folder = '/results/single/'

algorithm = None
problem = None
experiment = None

########

parser = argparse.ArgumentParser(description="Pass parameters for search.")
parser.add_argument('-e', dest='exp_number', type=str, action='store',
                    help='Hardcoded example scenario to use [2 to 6].')
parser.add_argument('-r', dest='n_runs', default=1, type=int, action='store',
                    help='Number of times to repeat experiment.')
parser.add_argument('-i', dest='n_generations', type=int, default=None, action='store',
                    help='Number generations to perform.')
parser.add_argument('-n', dest='size_population', type=int, default=None, action='store',
                    help='The size of the initial population of scenario candidates.')
parser.add_argument('-a', dest='algorithm', type=int, default=None, action='store',
                    help='The algorithm to use for search, 1 for NSGA2, 2 for NSGA2-DT.')
parser.add_argument('-t', dest='maximal_execution_time', type=str, default=None, action='store',
                    help='The time to use for search with nsga2-DT (actual search time can be above the threshold, since algorithm might perform nsga2 iterations, when time limit is already reached.')
parser.add_argument('-f', dest='scenario_path', type=str, action='store',
                    help='The path to the scenario description file/experiment.')
parser.add_argument('-min', dest='var_min', nargs="+", type=float, action='store',
                    help='The lower bound of each parameter.')
parser.add_argument('-max', dest='var_max', nargs="+", type=float, action='store',
                    help='The upper bound of each parameter.')
parser.add_argument('-m', dest='design_names', nargs="+", type=str, action='store',
                    help='The names of the variables to modify.')
parser.add_argument('-dt', dest='max_tree_iterations', type=int, action='store',
                    help='The maximum number of total decision tree generations (when using NSGA2-DT algoritm).')
parser.add_argument('-o', dest='results_folder', type=str, action='store', default=os.sep + "results" + os.sep,
                    help='The name of the folder where the results of the search are stored (default: \\results\\single\\)')
parser.add_argument('-s', dest='seed', type=int, default=None, action='store',
                help='Seed number to be used.')
args = parser.parse_args()

#######

if args.exp_number and args.scenario_path:
    log.info("Flags set not correctly: Experiment file and example experiment cannot be set at the same time")
    sys.exit()
elif not (args.exp_number or args.scenario_path):
    log.info("Flags set not correctly: No file is provided or no example experiment selected.")
    sys.exit()

###### set experiment
####### have indiviualized imports
if args.exp_number:
    # exp_number provided
    selExpNumber = re.findall("[0-9]+", args.exp_number)[-1]
    log.info(f"Selected experiment number: {selExpNumber}")
    experiment = experiment_switcher.get(int(selExpNumber))()

    config = experiment.search_configuration
    problem = experiment.problem
    algorithm = experiment.algorithm

elif (args.scenario_path):
    scenario_path = args.scenario_path
    var_min = []
    var_max = []

    #TODO create an experiment from user input
    #TODO create an ADASProblem from user input

    log.info("-- Experiment provided by file")

    if args.var_min is None:
        log.info("-- Minimal bounds for search are not set.")
        sys.exit()

    if args.var_max is None:
        log.info("-- Maximal bounds for search are not set.")
        sys.exit()

    log.info("Creating an experiment from user input not yet supported. Use default_experiments.py to create experiment")
    sys.exit()
else:
    log.info("-- No file provided and no experiment selected")
    sys.exit()

'''
override params if set by user
'''

if not args.size_population is None:
    config.population_size = args.size_population
if not args.n_generations is None:
    config.n_generations = args.n_generations
    config.inner_num_gen = args.n_generations #for NSGAII-DT
if not args.algorithm is None:
    algorithm = AlgorithmType(args.algorithm)
if not args.maximal_execution_time is None:
    config.maximal_execution_time = args.maximal_execution_time
if not args.max_tree_iterations is None:
    config.max_tree_iterations = args.max_tree_iterations
if not args.max_tree_iterations is None:
    results_folder = args.results_folder
if not args.var_max is None:
    problem.var_max = args.var_max
if not args.var_min is None:
    problem.var_min = args.var_min
if not args.design_names is None:
    problem.design_names = args.design_names
if args.seed is not None:
    problem.set_seed(args.seed)

####### Run algorithm

if __name__ == "__main__":
    execTime = None
    algo = None
    for i in range(args.n_runs):
        if algorithm == AlgorithmType.NSGAII:
            log.info("pymoo NSGA-II algorithm is used.")
            algo = NSGAII_SIM(
                                problem=problem,
                                config=config)
        elif algorithm == AlgorithmType.NSGAIIDT:
            log.info("NSGAII-DT algorithm is used.")
            algo = NSGAII_DT_SIM(
                                problem=problem,
                                config=config)
        elif algorithm == AlgorithmType.PS_RAND:
            log.info("pymoo PureSampling algorithm is used.")
            algo = PureSamplingRand(
                                problem=problem,
                                config=config)
        elif algorithm == AlgorithmType.PS_GRID:
            log.info("pymoo PureSampling algorithm is used.")
            algo = PureSamplingGrid(
                                problem=problem,
                                config=config)
        else:
            raise ValueError("Error: No algorithm with the given code: " + str(algorithm))
        
        res = algo.run()
        algo.write_results(results_folder=results_folder,
                ref_point_hv=None,
                nadir=None,
                ideal=None)

    log.info("====== Algorithm search time: " + str("%.2f" % res.exec_time) + " sec")
