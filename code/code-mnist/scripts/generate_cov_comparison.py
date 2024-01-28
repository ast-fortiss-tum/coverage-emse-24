from pathlib import Path
from algorithm.ps_rand import PureSamplingRand
from algorithm.nsga2_dt_sim import NSGAII_DT_SIM
from algorithm.nsga2_sim import NSGAII_SIM
from default_experiments import *
from experiment.search_configuration import *
import argparse
import re
from default_experiments import *
import logging as log
from algorithm_analysis.Analysis import Analysis
from problem.metric_config import MetricConfig
from utils.log_utils import *

logger = log.getLogger(__name__)

setup_logging("./analysis_log.txt")
disable_pymoo_warnings()
debug=False

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
    # exp_numbers_default = [5,3]  # Test
    
    ##### Dummy
    
    ################

    # 3: NLSGAII
    # 4: NSGAII_F
    # 5: NSGAII
    # 7: LNSGAII
    # 9: RS
    
    exp_numbers_default = [
        5,
        # 4,
        # 6,
        # 7,
        9
    ]  # Test

    ############### Specify the algorithms

    class_algos = [
        NSGAII_SIM,
        # NSGAII_SIM,
        # NLSGAII_SIM,
        #LNSGAII_SIM,
        PureSamplingRand
    ]

    ################ For Visualization

    algo_names = [
        "NSGA-II",
        # "NSGA-II-F",
        #"NLSGA-II",
         #"LNSGA-II",
         "RS"
   
    ]
    #############################
     
    n_runs_default = 10
    analyse_runs = n_runs_default

    ########## HV config #############
    # AVP
    # ref_point_hv = np.asarray([-0.6,-0.1])
    # ideal = np.asarray([-1,-4])

    # MNIST
    # Distance
    # ref_point_hv = np.asarray([-0.2,-1])
    # ideal = np.asarray([-1,-10])
    
    # Coverage
    ref_point_hv = MetricConfig.config["MNIST"]["ref_point_hv"]
    ideal = MetricConfig.config["MNIST"]["ideal"]

    # Dummy
    # ref_point_hv = np.asarray([4,0])
    # ideal = np.asarray([0,-10])

    distance_tick = 200

    nadir = ref_point_hv

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

    config_1 = DefaultSearchConfiguration()
    config_1.population_size = 50
    config_1.n_generations = 20

    ####################

    config_2 = DefaultSearchConfiguration()
    config_2.population_size =    1000
    config_2.n_generations =  config_1.n_generations 
    configs = [
               config_1,
               config_2
    ]
    
    n_func_evals_lim = 1000
    # this variable is require by an analysis function; TODO refactor 

    ################ Naming
    analysis_name = None

    output_folder = None
    folder_runs = None

    path_critical_set = r"/home/sorokin/Projects/testing/foceta/SBT-research/results/single/MNIST_GS/RS/temp/all_critical_testcases.csv"
    
    #######################

    problem = "MNIST_NLSGA-II"
    f_nruns = "2_runs"
    date = "02-11-2023_23-54-12"

    path = r"/home/sorokin/Projects/testing/foceta/SBT-research/results/analysis/MNIST_RS/10_runs/05-11-2023_18-30-33/"
    folder_runs = [
        path,
        path
    ]

    #folder_runs = None


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
                do_coverage_analysis=True,
                do_ds_analysis=True,
                path_critical_set=path_critical_set,
                debug=debug,
                distance_tick=distance_tick,
                do_evaluation=True
    )

