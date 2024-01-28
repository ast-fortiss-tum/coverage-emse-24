import pymoo

from model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from model_ga.result  import ResultExtended
pymoo.core.result.Result = ResultExtended

from model_ga.problem import ProblemExtended
pymoo.core.problem.Problem = ProblemExtended

from pathlib import Path
from algorithm import nsga2_dt_sim
from algorithm.lnsga2_sim import LNSGAII_SIM
from algorithm.nlsga2_sim import NLSGAII_SIM
from algorithm.nsga2_dt_sim import NSGAII_DT_SIM
from algorithm.nsga2_sim import NSGAII_SIM
from default_experiments import *
from experiment.search_configuration import *
import argparse
import re
from default_experiments import *
import logging as log
from algorithm_analysis.Analysis import Analysis
from utils.log_utils import *
from config import *

disable_pymoo_warnings()
debug=DEBUG

if __name__ == "__main__":        

    parser = argparse.ArgumentParser(description="Pass parameters for analysis.")
    parser.add_argument('-r', dest='n_runs', type=int, default=None, action='store',
                        help='Number runs to perform each algorithm for statistical analysis.')
    parser.add_argument('-p', dest='folder_runs',nargs="+", type=str, default=None, action='store',
                        help='The folder of the results written after executed runs of both algorithms. Path needs to end with "/".')
    parser.add_argument('-e', dest='exp_number', type=str, action='store',
                        help='Hardcoded example experiment to use.')
    parser.add_argument('-c', dest='path_metrics', type=str, default=None, action='store',
                        help='Path to csv file with metric results to regenerate comparison plot')
    
    args = parser.parse_args()
    
    ############# Set default experiment

    # we need to pass several exps, to be able to compare searches with different fitnessfnc 
    # (TODO check if fitness func should be part of a problem)
    # If the problem is the same, just pass the experiment number twice
        
    # exp_number_default =  1 # bnh
    # exp_numbers_default = [19,20]  # Filtered NSGAII
    # exp_numbers_default = [100,101,100]  # Test
    
    ##### Dummy
    # exp_numbers_default = [78,79]  # Test

    ##### AVP
    exp_numbers_default = [100,
                            101]

    ############### Specify the algorithms

    class_algos = [
        NSGAII_SIM,
        NSGAII_DT_SIM
    ]

    ################ For Visualization

    algo_names = [
        "NSGA-II",
        "NSGA-II-DT"

    ]
    #############################
     
    n_runs_default = 10
    
    n_func_evals_lim = 100# this variable is required by an analysis function; TODO refactor 
    analyse_runs = n_runs_default

    distance_tick = 0.1*n_func_evals_lim

    if args.exp_number is None:
        exp_numbers = exp_numbers_default
    else:
        exp_numbers = [re.findall("[1-9]+", exp)[0] for exp in args.exp_number]

    if args.n_runs is None:
        n_runs = n_runs_default
    else:
        n_runs = args.n_runs

    folder_runs =  args.folder_runs
    path_metrics = args.path_metrics

    ###################
    problems = []
    configs = []
    for exp_n in exp_numbers:
        exp = experiment_switcher.get(int(exp_n))()
        problem = exp.problem
        problems.append(problem)
        configs.append(exp.search_configuration)

    ##################### Override config
    DO_OVERRIDE_CONFIG = False

    if DO_OVERRIDE_CONFIG:
        config_1 = DefaultSearchConfiguration()
        config_1.population_size = 50
        config_1.n_generations = 20

        ####################

        config_2 = DefaultSearchConfiguration()
        config_2.population_size = 20
        config_2.n_generations = 10
        #config_2.ref_points = np.asarray([[0,-4]])
        config_2.inner_num_gen = 5
        config_2.n_func_evals_lim = n_func_evals_lim
        configs = [config_2,config_1]

    # Math BNH
    ideal = configs[0].ideal
    ref_point_hv = configs[0].ref_point_hv
    nadir =  configs[0].nadir

    ################ Naming
    analysis_name = None

    ############# ONLY EVALUATE #######################
    
    output_folder = None
    #output_folder = os.getcwd() + os.sep + "results" + os.sep + "output" + os.sep

    #######################
    folder_runs = None

    # Use different critical funciton for evaluation (only for algos that don't use crit function for search)  
    crit_function = None
    folder_runs = None

    Analysis.run(
                analysis_name=analysis_name,
                algo_names = algo_names,
                class_algos = class_algos,
                configs = configs,
                n_runs = n_runs,
                problems = problems,
                n_func_evals_lim = n_func_evals_lim, 
                folder_runs = folder_runs,
                path_metrics = path_metrics,
                ref_point_hv=ref_point_hv,
                ideal=ideal,
                nadir=nadir,
                output_folder=output_folder,
                do_coverage_analysis=False,
                do_ds_analysis=True,
                path_critical_set=None,
                debug=debug,
                distance_tick=distance_tick,
                do_evaluation=True,
                crit_function = crit_function
    )
