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
from config import *

logger = log.getLogger(__name__)
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
    parser.add_argument('-s', dest='seed', type=int, default=None, action='store',
                    help='Seed number to be used.')


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
    
    exp_numbers_default = [
        9,
        5,   
        6
    ]

    ############### Specify the algorithms

    class_algos = [
        PureSamplingRand,
        NSGAII_SIM,
        NSGAII_SIM
    ]

    ################ For Visualization

    algo_names = [
        "RS",
        "NSGA-II-D", 
        "NSGA-II"
        #"NSGA-II-D"
    ]
    #############################
     
    n_runs_default = 1
    analyse_runs = n_runs_default

    ########## HV config #############
    p_name = "MNIST_MULTI"
    #p_name = "MNIST"

    ref_point_hv = MetricConfig.config[p_name]["ref_point_hv"]
    ideal = MetricConfig.config[p_name]["ideal"]

    n_func_evals_lim = 1000
    
    distance_tick = n_func_evals_lim * 0.10

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

        # change seed
        if args.seed is not None:
            problem.set_seed(args.seed)

        problems.append(problem)
        configs.append(exp.search_configuration)
    ##################### Override config
    OVERRIDE_CONFIG = True

    if OVERRIDE_CONFIG:
        config_rs = DefaultSearchConfiguration()
        config_rs.population_size = 1000

        config_1 = DefaultSearchConfiguration()
        config_1.population_size = 100
        config_1.n_generations = 10

        ####################

        config_2 = DefaultSearchConfiguration()
        config_2.population_size =    100
        config_2.n_generations =  config_1.n_generations 
        #config_2.ref_points = np.asarray([[0,-4]])
        # config_2.inner_num_gen = 5
        # config_2.n_func_evals_lim = 2000

        configs = [
                config_rs,
                config_1,
                config_1
                ]

    # this variable is require by an analysis function; TODO refactor 

    ################ Naming
    analysis_name = None

    ############# ONLY EVALUATE #######################

    # combined_analysis is written here
    output_folder = None
    folder_runs = None
    
    #######################
    path_critical_set = None #PATH_CRITICAL_SET
    #"/home/sorokin/Projects/testing/foceta/SBT-research/results/single/MNIST_GS/RS/mut_simple_132/all_critical_testcases.csv"
    
    path = r'/home/sorokin/Projects/testing/foceta/SBT-research/results/analysis/MNIST_3D_129_NSGA-II-D/10_runs/23-11-2023_01-26-48/'
    #'/home/sorokin/Projects/testing/foceta/SBT-research/results/analysis/MNIST_3D_129_NSGA-II-D/10_runs/19-11-2023_23-00-16/'
    # # path = "/home/sorokin/Projects/testing/foceta/SBT-research/results/analysis/MNIST_NLSGA-II/15_runs/16-11-2023_17-31-07/"
    folder_runs = [
        path,
        path,
        path
    ]
    folder_runs = None

    crits = [
        None
    ]
    
    ######## Analysis using different oracle functions (only if folder_runs not None) 

    crit_function_07 = CriticalMNISTConf_07()
    crit_function_05 = CriticalMNISTConf_05()
    crit_function_095 = CriticalMNISTConf_095()
    
    #crit_function_09 = CriticalMNISTConf_09()
    # crits = [crit_function_095, 
    #         crit_function_05,
    #         crit_function_07]
   
    #crits = [CriticalMNISTConf_05]
    ##########################

    n_evals_by_axis = 15

    for crit_function in crits:     
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
                    do_ds_analysis=False,
                    path_critical_set=path_critical_set,
                    debug=debug,
                    distance_tick=distance_tick,
                    do_evaluation=True,
                    crit_function=crit_function,
                    n_evals_by_axis=n_evals_by_axis,
                     analysis_parent_folder="results" + os.sep + "analysis" + os.sep + "multiseed"
        )

