# from simulation.prescan_simulation import PrescanSimulator
import pymoo

from model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from model_ga.result  import ResultExtended
pymoo.core.result.Result = ResultExtended

from model_ga.problem import ProblemExtended
pymoo.core.problem.Problem = ProblemExtended

import dill
from visualization.combined import write_last_metric_values
import traceback
from utils.sampling import cartesian_reference_set
import signal
from pympler import asizeof
import gc
import psutil
import time
import logging as log
from utils.path import get_subfolders_from_folder
import re
import argparse
import sys
from experiment.search_configuration import *
from visualization import output
from pathlib import Path
from datetime import datetime
import os
from visualization.combined import *
from default_experiments import *
from algorithm.ps_rand import PureSamplingRand
from algorithm.nsga2_sim import NSGAII_SIM
from utils.analysis_utils import read_critical_set
from visualization.combined import statistical_analysis
from visualization import configuration
from config import *
from utils import log_utils
from visualization import output_mnist
from utils import file_utils

''' This script works in two modes:
    1. Repeated Executions of experiments  + Analysis of Results (Applying Metrics) 
    2. Analysis of Results 

    For 1) just configure the algorithms, their config, n_runs (bottom part of this file)
    
    For CID analysis the estimated critical set needs to be pre-computed and past through the PATH_CRITICAL_SET variable.

    For 2) just run analysis.py and pass via -p the path to stored experiment results. 
'''

''' 
    Reference sets for SSBSE23 Paper
'''


class Analysis(object):
    # Flags for restarts
    # DO_COVERAGE_ANALYSIS = False
   
    REPEAT_RUN = True
    MAX_REPEAT_FAILURE = 50
    TIME_WAIT = 10  # for restart in sec

    @staticmethod
    def run(analysis_name,
            class_algos,
            configs,
            n_runs,
            problems,
            n_func_evals_lim,
            folder_runs=None,
            path_metrics=None,
            n_select=None,
            n_fitting_points=8,
            distance_tick=None,
            ref_point_hv=None,
            do_coverage_analysis=False,
            do_ds_analysis=True,
            ideal=None,
            nadir=None,
            algo_names=None,
            output_folder=None,
            path_critical_set=None,
            debug=False,
            do_evaluation=True,
            crit_function=None,
            n_evals_by_axis=10,
            analysis_parent_folder=ANALYSIS_PARENT_FOLDER,
            title_plot=None
        ):
                
        # the number of algorithms analysed
        n_algos = len(class_algos)
        
        if algo_names is None:
            log.info("Algo names is none")
            algo_names = []
            for i, cl in enumerate(class_algos):
                algo_names.append(cl.algorithm_name)

        run_paths_all = {}
        analysis_folder = None

        if analysis_name is None:
            analysis_name = problems[0].problem_name
            log.info("Using problem name as analysis name.")

        # if no output folder provided, write analysis results in folder of first experiment;
        # Output folder is ONLY used for writing combined analysis results

        if output_folder is None:
            if folder_runs is not None and len(folder_runs) > 1:
                output_folder = folder_runs[0] + \
                    f"comparison_{algo_names}" + os.sep
                Path(output_folder).mkdir(parents=True, exist_ok=True)

            if crit_function is not None:
                # Use class name to distinguish between different outputs
                suffix = f"_{crit_function.__class__.__name__}"
                
                # If a criticality function is passed, regenerate results object for different criticality and evaluate later
                algo_paths = []

                for i, algo in enumerate(algo_names):
                    algo_paths.append(folder_runs[i] + algo + os.sep)

                out_parent_folder = analysis_parent_folder + suffix + os.sep + os.path.relpath(folder_runs[i],analysis_parent_folder) + os.sep
                
                # terminate if folder already exists
                if os.path.exists(out_parent_folder):
                    print(f"Skipping generation of modifed results because analysis path {out_parent_folder} already exists.")
                    output_folder = out_parent_folder                
                else:
                    Path(out_parent_folder).mkdir(parents=True, exist_ok=True)

                    print(f"++++++ out_parent_folder: {out_parent_folder}")
                    output_folder = Analysis.generate_modified_results(
                        algo_paths=algo_paths,
                        crit_fnc=crit_function,
                        suffix=suffix,
                        out_parent_folder=out_parent_folder)
                    # output_folder = os.getcwd() + os.sep + "test"

                    Path(output_folder).mkdir(parents=True, exist_ok=True)

            # else:
            #     output_folder = folder_runs[0]
            #     # we use the same folder for the analysis
            #     folder_runs.append(folder_runs[0])
        else:
            Path(output_folder).mkdir(parents=True, exist_ok=True)

        if (folder_runs is not None) and (path_metrics is not None):
            analysis_folder = folder_runs[0]
            log.info("Regenerationg comparison plot from given anaylsis data.")
            Analysis.regenerate_comparison_plot(algo_names, path_metrics, analysis_folder,
                                                n_func_evals_lim, n_fitting_points=n_fitting_points, distance_tick=distance_tick)
            log.info("Regeneration completed.")
        else:
            if (folder_runs is not None):
                ''' provide folder of finished runs to do analysis '''
                log.info("Loading data from completed runs.")

                for i, algo_name in enumerate(algo_names):
                    if crit_function is not None:
                        run_paths_all[algo_name] = get_subfolders_from_folder(
                            output_folder + algo_name + os.sep)
                    else: 
                        run_paths_all[algo_name] = get_subfolders_from_folder(
                            folder_runs[i] + algo_name + os.sep)

                # we can select subset of runs to use for evaluation
                if n_select is not None:
                    for algo_name in algo_names:
                        run_paths_all[algo_name] = run_paths_all[algo_name][1:(
                            n_select+1)]

                # analysis_folder = folder_runs
                log_utils.setup_logging(output_folder + os.sep + "log.txt")


                log.info(
                    f"Analysed algorithms: {list(run_paths_all.keys())} \n")
                
            else:
                log.info("Executing algorithms for analysis.")
                if debug:
                    analysis_folder = str(Path(os.getcwd()).joinpath(
                       analysis_parent_folder, analysis_name, f"{n_runs}_runs", "temp")) + os.sep
                else:
                    analysis_folder = str(Path(os.getcwd()).joinpath(
                       analysis_parent_folder, analysis_name, f"{n_runs}_runs", datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))) + os.sep

                Path(analysis_folder).mkdir(parents=True, exist_ok=True)

                log_utils.setup_logging(analysis_folder + "./log.txt")
                
                ''' run the algorithms first and then do analysis '''
                # results are written in results/analysis/<problem>/<n_runs>/<date>/
                for i, config in enumerate(configs):
                    log.info(f"Search config {i}: {config.__dict__} \n")
                run_paths_all = Analysis.execute_algos(
                    configs, problems, class_algos, algo_names, n_runs, analysis_folder)

            log.info("folder_runs is: " + str(folder_runs))
            
            if do_evaluation:             
                if analysis_folder is not None:
                    output_folder =  analysis_folder
                
                log.info("Evaluating runs...")
                log.info("output folder is: " + str(output_folder))

                Analysis.evaluate_runs(algo_names,
                                    run_paths_all,
                                    output_folder,
                                    n_func_evals_lim,
                                    n_fitting_points=n_fitting_points,
                                    distance_tick=distance_tick,
                                    ref_point_hv=ref_point_hv,
                                    problems=problems,
                                    do_ds_analysis=do_ds_analysis,
                                    do_coverage_analysis=do_coverage_analysis,
                                    ideal=ideal,
                                    nadir=nadir,
                                    path_critical_set=path_critical_set,
                                    n_evals_by_axis=n_evals_by_axis,
                                    critical_fnc=crit_function,
                                    title_plot=title_plot)

    @staticmethod
    def get_memory_load():
        return psutil.Process().memory_info().rss / (1024 * 1024)

    @staticmethod
    def create_run_folder(analysis_folder, algorithm, run_num):
        i = run_num
        run_folder = analysis_folder + \
            str(algorithm) + os.sep + str(f"run_{i}") + os.sep
        Path(run_folder).mkdir(parents=True, exist_ok=True)
        return run_folder

    @staticmethod
    def write_results_reduced(res, 
        results_folder, 
        algorithm_name, 
        algo_parameters,
        simdata_available=True):

        output.design_space(res, results_folder)
        output.objective_space(res, results_folder)
        output.optimal_individuals(res, results_folder)
        output.write_summary_results(res, results_folder)
        output.write_calculation_properties(res, results_folder,
                                            algorithm_name=algorithm_name,
                                            algorithm_parameters=algo_parameters)
        # output.simulations(res, results_folder)
        output.all_individuals(res, results_folder)
        output.all_critical_individuals(res, results_folder)
        output.write_generations(res, results_folder)
        # quality
        # output.hypervolume_analysis(res, results_folder)
        # output.spread_analysis(res, results_folder)
        # experimental
        # output.spread_analysis_hitherto(res, results_folder)
        if simdata_available:
            from problem.mnist_problem import MNISTProblem
            if type(res.problem) == MNISTProblem:
                print("Exporting inputs ...")
                # output_mnist.output_optimal_digits(res, save_folder)
                # output_mnist.output_explored_digits(res, save_folder)
                # output_mnist.output_critical_digits(res, save_folder)
                output_mnist.output_critical_digits_all(res, results_folder)
                # output_mnist.output_seed_digits(res, save_folder)
                output_mnist.output_optimal_digits_all(res, results_folder)
                output_mnist.output_seed_digits_all(res, results_folder)
                # output_mnist.write_generations_digit(res, results_folder)
                # output_mnist.output_summary(res, save_folder)

    @staticmethod
    def execute_algos(configs, problems, class_algo, algo_names, n_runs, analysis_folder):
        # make robust against unintended exits
        def handler(signum, frame):
            res = input("Ctrl-c was pressed. Do you really want to exit? y/n ")
            if res == 'y':
                log.info("Terminating program...")
                sys.exit(1)
        signal.signal(signal.SIGINT, handler)

        run_paths_all = {}
        for i, _ in enumerate(class_algo):
            run_paths_all[algo_names[i]] = []

        for j, algo_name in enumerate(algo_names):
            problem = problems[j]
            for i in range(1, n_runs+1):
                do_repeat = True
                cnt = 0
                while (do_repeat and cnt <= Analysis.MAX_REPEAT_FAILURE):
                    try:
                        log.info(
                            f"------ Running run {i} from {n_runs} with {algo_name} ------ \n")
                        run_folder = Analysis.create_run_folder(
                            analysis_folder, algo_name, i)
                        config = configs[j]
                        algo = class_algo[j](
                            problem=problem,
                            config=config)
                        res = algo.run()

                        do_repeat = False
         
                        if STORE_RESULT_OBJECT:
                            log.info("----- Storing result object ------")
                            log.info(f"-----Result object has size: {sys.getsizeof(res)/1000} kB")
                            res.persist(run_folder + BACKUP_FOLDER)

                        log.info("----- Storing problem object ------")
                        output.backup_problem(res, run_folder)

    
                        # log.info("----- Reduced writing of results ------")
                        Analysis.write_results_reduced(res,
                                                       run_folder,
                                                       algorithm_name=algo.algorithm_name,
                                                       algo_parameters=config.__dict__)

                        run_paths_all[algo_name].append(run_folder)

                        log.info(
                            f"---- Evaluating run {i} from {n_runs} with {algo_name} completed ----\n")

                        #### release memory

                        del res
                        del algo
                        del config

                        from problem.mnist_loader import mnist_loader
                        del mnist_loader

                        from problem.mnist.predictor import model
                        del model

                        gc.collect()

                        
                        # from problem.mnist_problem import MNISTProblem
                        # for i in range(1,100):
                        #     digit = MNISTProblem.generate_and_evaluate_digit(2)
                        #     digit.purified = None
                        plt.close('all')
                        
                        # import objgraph

                        # objgraph.show_growth()

                    except Exception as e:
                        traceback.print_exc()
                        if problem.is_simulation():
                            PrescanSimulator.kill()
                        gc.collect()
                        if Analysis.REPEAT_RUN:
                            log.error(
                                f"\n---- Repeating run {i} due to exception: ---- \n {e} \n")
                            time.sleep(Analysis.TIME_WAIT)
                            cnt = + 1
                        else:
                            do_repeat = False
        log.info("---- All runs completed. ---\n-")

        return run_paths_all

    @staticmethod
    def regenerate_comparison_plot(algo_names,
                                   paths_results_csv,
                                   output_folder,
                                   n_func_evals_lim,
                                   n_fitting_points,
                                   distance_tick):
        # temporary some params hard coded
        subplot_names = ["CID"]
        metric_data_loaded = retrieve_metric_data_from_csv([paths_results_csv])
        make_comparison_plot(n_func_evals_lim,
                             output_folder,
                             metric_data_loaded,
                             subplot_names,
                             algo_names,
                             distance_tick=distance_tick,
                             suffix="_ds")
        
    def generate_modified_results(algo_paths,
                                crit_fnc,
                                suffix,
                                out_parent_folder=None):
                
        def update_ind(population, crit_fnc):
            for ind in population:
                ind.set("CB", crit_fnc.eval(ind.get("F"), simout=None))
                ind.set("DIG", None)
                #ind.set("DIG", None) # delete simout to save storage
            return population

        for path in algo_paths:
            algo_name = os.path.basename(os.path.dirname(path))

            # create new parent folder
            if out_parent_folder is None: 
                date_path = os.path.dirname(os.path.dirname(path))
                output_folder =  date_path + suffix + os.sep
                output_path = output_folder +  algo_name + os.sep
            # use parent folder of paths
            else:
                Path(out_parent_folder).mkdir(parents=True, exist_ok=True)
                output_folder = out_parent_folder
                output_path = out_parent_folder +  algo_name + os.sep

            run_folders  = os.listdir(path)
            print(f"folder in path:{run_folders}")
            for run in run_folders:
                run_path = path + run + os.sep
                output_path_run = output_path + os.sep + run + os.sep
                # create results from generations
                log.info(f"[Analysis] Creating result object from generations...")
                path_generations = run_path + "generations" + os.sep

                with open(run_path + os.sep + "backup" + os.sep + "problem", "rb") as f:
                    problem = dill.load(f)
                res = output.create_result_from_generations(
                                        path_generations,
                                        problem=problem
                )
                problem = res.problem
                problem.critical_function = crit_fnc
                
                # modify CB values in history
                for algo in res.history:
                    update_ind(algo.pop, crit_fnc=crit_fnc)
                    update_ind(algo.opt, crit_fnc=crit_fnc)
                    algo.problem = problem

                update_ind(res.opt, crit_fnc=crit_fnc)
                res.persist(output_path_run + "backup" + os.sep)

                Analysis.write_results_reduced(res,
                    results_folder = output_path_run, 
                    algorithm_name = algo_name, 
                    algo_parameters=dict(),
                    simdata_available=False)
        return output_folder

    @staticmethod
    def evaluate_runs(algo_names,
                      run_paths_all,
                      output_folder,
                      n_func_evals_lim,
                      n_fitting_points,
                      distance_tick,
                      ref_point_hv,
                      problems,
                      do_coverage_analysis,
                      do_ds_analysis,
                      ideal,
                      nadir,
                      path_critical_set,
                      n_evals_by_axis,
                      critical_fnc,
                      title_plot):

        log.info("############# Analysis #############")
        # algo_name_1 = algo_names[0]
        # algo_name_2 = algo_names[1]

        # select for now first probl<em
        problem = problems[0]

        if do_ds_analysis:
            # Analysis
            # Real pareto front is known
            pf_true = None
            if type(problem) is not ADASProblem:
                pf_true = problem.pareto_front_n_points()

            # Estimate pareto front by aggregating run results
            paths_all = []
            for algo in algo_names:
                paths_all += run_paths_all[algo]
            # calculate estimated pareto front
            log.info("---- Calculating estimated pareto front. ----\n")

            pf_estimated, pf_pop = calculate_combined_pf(paths_all, critical_only=True)

            #Write down estimated individuals of estimated pf
            output.write_pf_individuals(save_folder=output_folder,
                                        pf_pop=pf_pop)
            #  

        if do_coverage_analysis:
            from os.path import dirname

            # get parent folder of runs for on algo to look for ref set (temp hack walking in dir up)
            parent = dirname(dirname(dirname(run_paths_all[algo_names[0]][0])))
            if path_critical_set is not None:
                log.info("---- Reading approximated critical solutions set. ----\n")

                cs_estimated = read_critical_set(path_critical_set)

            elif (file_utils.find_file_in_subdirectory(parent, "reference_set.csv")):
                path_critical_set = file_utils.find_file_in_subdirectory(parent, "reference_set.csv")
                cs_estimated = read_critical_set(path_critical_set)
                print(f"reference_sef found: {path_critical_set}")
            else:
                log.info("---- Generating approximated critical solutions set. ----\n")
                cs_estimated = cartesian_reference_set(
                    problem, n_evals_by_axis=n_evals_by_axis)
        
            # Apply critical function if given on reference set
            if critical_fnc is not None:
                from utils.population import update_ind
                update_ind(cs_estimated, 
                            critical_fnc)
            # make sure only critical inds of reference set are considered
            cs_estimated = cs_estimated.divide_critical_non_critical()[0]
        
            #write down estimated critical set
            log.info("---- Writing approximated critical solutions set. ----\n")
            
            ref_set_folder = output_folder + os.sep + "ref_set" + os.sep
            Path(ref_set_folder).mkdir(parents=True, exist_ok=True)

            output.write_critical_tests(save_folder=ref_set_folder,
                                        pop=Population(individuals=cs_estimated),
                                        file_name="reference_set.csv")

            visualize_3d(population=Population(individuals=cs_estimated),
                            save_folder=ref_set_folder,
                            do_save=True,
                            labels=problem.design_names)

        # calculate estimated set of critical solutions
        # critical_all_algo1 = calculate_combined_crit_pop(run_paths_all[algo_name_1])
        # critical_all_algo2 = calculate_combined_crit_pop(run_paths_all[algo_name_2])

        # critical_all = Population.merge(critical_all_algo1,critical_all_algo2)
        # log.info(f"estimated pf: {pf_estimated}")
        # perform igd analysis/create plots

        # result_runs_all = { algo_name_1 : [],
        #                     algo_name_2 : []}

        for algo_name in algo_names:
            for run_path in run_paths_all[algo_name]:
                backup_path = run_path + BACKUP_FOLDER
                
                if LOAD_FROM_GENERATIONS:
                    log.info(f"[Analysis] Creating result object from generations...")

                    path_generations = run_path + "generations" + os.sep

                    res = output.create_result_from_generations(
                                            path_generations,
                                            problem=problem
                    )
                else:
                    ###################### load result
                    log.info(
                        f"[Analysis] Reading result object from {backup_path}")

                    with open(backup_path + "result", "rb") as f:
                        res = dill.load(f)
                        
                ##############################


                # Unccoment following if "write_analysis_results" has to performed later. Release of memory is then not possible
                # result_runs_all[algo_name].append(res)

                if do_ds_analysis:
                    if pf_estimated is not None:
                        log.info(
                            f"---- Size of estimated PF: {len(pf_estimated)}")

                    if pf_true is not None:
                        # output.gd_analysis(res, run_path, input_pf=pf_true, filename='gd_true')
                        output.gd_analysis(
                            res, run_path, input_pf=pf_true, critical_only=True, filename='gd_true')
                        output.igd_analysis(
                            res, run_path, input_pf=pf_true, critical_only=True, filename='igd_true')

                    output.gd_analysis(
                        res, run_path, input_pf=pf_estimated, critical_only=True, filename='gd')
                    output.igd_analysis(
                        res, run_path, input_pf=pf_estimated, critical_only=True, filename='igd')
                    output.spread_analysis(res, 
                                        run_path)
                    # output.si_analysis(res, 
                    #                    run_path,
                    #                    input_pf=pf_estimated,
                    #                    critical_only=True,
                    #                    nadir=nadir,
                    #                    ideal=ideal)

                    output.hypervolume_analysis(res,
                                                run_path,
                                                critical_only=True,
                                                ref_point_hv=ref_point_hv,
                                                ideal=ideal,
                                                nadir=nadir)              

                if do_coverage_analysis:
                    output.cid_analysis_hitherto(
                        res, reference_set=cs_estimated, save_folder=run_path)

                # release memory
                del res

                # Test
                # critical_pop_alg = calculate_combined_crit_pop(run_paths=[run_path])
                # output_temp.plot_critical_all(problem, population=critical_pop_alg, save_folder = run_path + os.sep + "temp" + os.sep)

        # create combined criticality plots
        # output_temp.design_space(problem, population=critical_all_algo1, save_folder = analysis_folder + os.sep + "critical_set" + os.sep, suffix=f"_{algo_name_1}", classification_type=None)
        # output_temp.design_space(problem, population=critical_all_algo2, save_folder = analysis_folder + os.sep + "critical_set" + os.sep, suffix=f"_{algo_name_2}", classification_type=None)
        # output_temp.design_space(problem, population=critical_all, save_folder = analysis_folder + os.sep + "critical_set" + os.sep, suffix="_all", classification_type=None)

        # Objective Space metrics

        # TODO refactor, beautify, make more modular/generic
        if do_ds_analysis:
            log.info(f"run_paths_all: {len(run_paths_all)}")
            log.info(f"output folder is: {output_folder}")

            # statistical tests
            metric_names = ['hv', 'gd', 'igd', 'sp']
            metric_names_loaded = ['hv_global', 'gd', 'igd', 'sp']

            for m_name, m_load in zip(metric_names,metric_names_loaded):
                write_last_metric_values(m_load, 
                                         run_paths_all, 
                                         output_folder + m_name + os.sep, 
                                         metric_name_label=m_name)

                statistical_analysis_from_overview(m_name, 
                                        input_folder = output_folder + m_name + os.sep,
                                        save_folder= output_folder + m_name + os.sep)

                # statistical_analysis(metric_name_load=m_load, 
                #                         runs_bases= run_paths_all,
                #                         runs_test = run_paths_all[algo_names[0]],
                #                         algo_test = algo_names[0] , # the first algorithm is the subject
                #                         save_folder=output_folder + m_name + os.sep,
                #                         metric_name_label=m_name)
                
            # combined plots
            plot_array_hv = plot_combined_analysis('hv_global',
                                                   run_paths_all,
                                                   output_folder + "hv" + os.sep, 
                                                   n_func_evals_lim,
                                                   n_fitting_points,
                                                   step_chkp=distance_tick)

            plot_array_sp = plot_combined_analysis(
                'sp', run_paths_all, output_folder + "sp" + os.sep, n_func_evals_lim, n_fitting_points, 
                step_chkp=distance_tick)
            plot_array_igd = plot_combined_analysis(
                'igd', run_paths_all, output_folder + "igd" + os.sep, n_func_evals_lim, n_fitting_points,
                step_chkp=distance_tick)
            plot_array_gd = plot_combined_analysis(
                'gd', run_paths_all, output_folder + "gd" + os.sep, n_func_evals_lim, n_fitting_points,
                step_chkp=distance_tick)

            # log.info(f"plot_array_hv: {plot_array_hv})")

            metric_names = ['hv', 'gd', 'sp']

            paths = write_metric_data_to_csv(
                output_folder, 
                metric_names, 
                algo_names, 
                plot_array_hv, 
                plot_array_gd, 
                plot_array_sp)
            
            metric_names = ['hv', 'igd', 'sp']

            paths = write_metric_data_to_csv(
                output_folder, 
                metric_names, 
                algo_names, 
                plot_array_hv, 
                plot_array_igd, 
                plot_array_sp,
                suffix="_igd")
        
            # plot_array_hv_loaded, plot_array_gd_loaded, plot_array_sp_loaded = retrieve_metric_data_from_csv(paths)
            subplot_metrics = [plot_array_hv,
                               plot_array_gd,
                               plot_array_sp]

            subplot_names = ["HV_C",
                             "GD_C",
                             "Spread_C"]
            # with gd
            make_comparison_plot(n_func_evals_lim,
                                 output_folder,
                                 subplot_metrics,
                                 subplot_names,
                                 algo_names,
                                 distance_tick=distance_tick,
                                suffix="_gd",
                                cmap=configuration.c_map)
    
            subplot_names = ["HV_C",
                        "IGD_C",
                        "Spread_C"]
            subplot_metrics = [plot_array_hv,
                            plot_array_igd,
                            plot_array_sp]
            # with igd
            make_comparison_plot(n_func_evals_lim,
                                 output_folder,
                                 subplot_metrics,
                                 subplot_names,
                                 algo_names,
                                 distance_tick=distance_tick,
                                 shift_error=True,
                                suffix="_igd",
                                 cmap=configuration.c_map)


        # Input Space Metrics/ Coverage Metric
        if do_coverage_analysis:
            path_coverage_results = output_folder + "cid" + os.sep
            plot_array_cid = plot_combined_analysis(
                "cid", 
                run_paths_all,
                path_coverage_results, 
                n_func_evals_lim, 
                n_fitting_points, 
                COVERAGE_METRIC_NAME, 
                step_chkp=distance_tick)

            # subplot_metrics = [plot_array_cid]

            subplot_names = ["CID"]*3

            metric_names = ['cid', 'cid', 'cid']

            paths = write_metric_data_to_csv(
                        output_folder, 
                        metric_names, 
                        algo_names, 
                        plot_array_cid, 
                        plot_array_cid, 
                        plot_array_cid,
                        suffix="")

            subplot_metrics = [plot_array_cid]
            #metric_data_loaded = retrieve_metric_data_from_csv(paths)
            make_comparison_plot(n_func_evals_lim, 
                                 output_folder, 
                                 subplot_metrics,
                                 subplot_names, 
                                 algo_names, 
                                 shift_error=False,
                                 distance_tick=distance_tick, 
                                 suffix="_ds",
                                 title_plot=title_plot)

            write_last_metric_values(
                "cid", run_paths_all, path_coverage_results, "cid")
            m_name = 'cid'
            statistical_analysis_from_overview(m_name, 
                                    input_folder = output_folder + m_name + os.sep,
                                    save_folder= output_folder + m_name + os.sep)
        ###############

        log.info("---- Analysis plots generated. ----")

        # write_analysis_results(result_runs_all, analysis_folder)

        log.info("---- Analysis summary written to file.")
        # log.info(f"---- Current memory load: {get_memory_load()} MB")
        ######################

        log.info(f"Results written in: {output_folder}")
