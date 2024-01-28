
import pymoo
from evaluation.critical import *

# from model_ga.individual import IndividualSimulated
# pymoo.core.individual.Individual = IndividualSimulated

# from model_ga.population import PopulationExtended
# pymoo.core.population.Population = PopulationExtended

# from model_ga.result  import ResultExtended
# pymoo.core.result.Result = ResultExtended

# from model_ga.problem import ProblemExtended
# pymoo.core.problem.Problem = ProblemExtended

import copy
import sys
import time

from pymoo.core.result import Result
from pymoo.termination import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from pymoo.core.problem import Problem
from exception.configuration import RestrictiveConfigException
from visualization import output
from problem.adas_problem import ADASProblem
from problem.pymoo_test_problem import PymooTestProblem
from utils.time_utils import convert_pymoo_time_to_seconds
from pymoo.core.population import Population
from visualization.configuration import *
from algorithm.classification.classifier import ClassificationType
from algorithm.classification.decision_tree import decision_tree
from utils.evaluation import evaluate_individuals
from utils.sorting import *
from utils.time_utils import convert_pymoo_time_to_seconds
from visualization.configuration import *
import numpy as np
import os
from experiment.search_configuration import DefaultSearchConfiguration, SearchConfiguration
import quality_indicators.metrics.spread as qi
from model_ga.result import *
import logging as log
from quality_indicators.quality import *
from visualization import output_mnist
from config import *

class NSGAII_DT_SIM(object):

    algorithm_name = "NSGA-II-DT"

    def __init__(self,
                 problem: Problem,
                 config: SearchConfiguration):

        self.problem = problem
        self.config = config
        self.done_tree_iterations = 0
        self.res = None

        log.info(f"Initialized algorithm with config: {config.__dict__}")
        
    def run(self) -> ResultExtended:
        problem = self.problem
        config = self.config

        population_size = config.population_size
        maximal_execution_time = config.maximal_execution_time
        max_tree_iterations = config.max_tree_iterations
        num_offsprings = config.num_offsprings
        prob_crossover = config.prob_crossover
        eta_crossover = config.eta_crossover
        prob_mutation = config.prob_mutation
        eta_mutation = config.eta_mutation
        inner_num_gen = config.inner_num_gen

        '''Output variables'''
        all_population = Population()
        best_population = Population()
        best_population_unranked = Population()

        '''Initial conditions (initial region)'''
        xl = problem.xl
        xu = problem.xu

        # sampling = FloatRandomSampling()
        sampling = LHS()  # Latin Hypercube Sampling
        initial_population = sampling(problem, population_size)
        hist_holder = []

        '''Parameters of the algorithm'''
        if prob_mutation is None:
            prob_mutation = 1 / problem.n_var

        '''Parameter for evaluation'''
        if maximal_execution_time is not None:
            _maximal_execution_time = convert_pymoo_time_to_seconds(
                maximal_execution_time)
            max_tree_iterations = sys.maxsize
            log.info("Search is constrained by maximal execution time")
        elif max_tree_iterations is not None:
            _maximal_execution_time = sys.maxsize
            log.info("Search is constrained by maximal number of tree generations")
        else:
            log.info("Parameters are not correctly set, cannot start search.")
            sys.exit()

        ''' Computation start '''
        start_time = time.time()

        evaluate_individuals(initial_population, problem)
        initial_region = decision_tree.Region(xl, xu, initial_population)
        critical_regions = [initial_region]
        hist_holder = []

        # inner_algorithm is a template for an algorithm object that is stored for every generation
        inner_algorithm = NSGA2(
            pop_size=None,
            n_offsprings=None,
            sampling=None,
            crossover=SBX(prob=prob_crossover, eta=eta_crossover),
            mutation=PM(prob=prob_mutation, eta=eta_mutation),
            eliminate_duplicates=True)

        tree_iteration = 0 
        n_func_evals = 0
        while n_func_evals < config.n_func_evals_lim:
            # extend the history by one generation
            hist_holder.extend([inner_algorithm] * inner_num_gen)

            log.info(f"running iteration {tree_iteration}")
            for critical_region in critical_regions:
                sub_problem = problem

                if prob_mutation == None:
                    prob_mutation = 1 / problem.n_var
                    
                nd_individuals_region = calc_nondominated_individuals(
                    critical_region.population)
                initial_population = Population(
                    individuals=nd_individuals_region)
                pop_size = len(initial_population)

                if self.config.prob_mutation is None:
                    self.config.prob_mutation = 1 / problem.n_var

                if config.operators is not None and config.operators["init"]:
                    sampling = config.operators["init"]()
                else:
                    sampling = FloatRandomSampling()
                
                if config.operators is not None and config.operators["cx"]:
                    crossover = config.operators["cx"]()
                else:
                    crossover = SBX(prob=config.prob_crossover, eta=config.eta_crossover)

                if config.operators is not None and config.operators["mut"]:
                    mutation = config.operators["mut"]()
                else:
                    mutation = PM(prob=config.prob_mutation, eta=config.eta_mutation)
                
                if config.operators is not None and config.operators["dup"]:
                    eliminate_duplicates = config.operators["dup"]()
                else:
                    eliminate_duplicates = True

                algorithm = NSGA2(
                    pop_size=pop_size,
                    n_offsprings=config.num_offsprings,
                    sampling=sampling,
                    crossover=crossover,
                    mutation=mutation,
                    eliminate_duplicates=eliminate_duplicates)

                termination = get_termination("n_gen", inner_num_gen)

                res = minimize(sub_problem,
                               algorithm,
                               termination,
                               seed=1,
                               save_history=True,
                               verbose=False)

                n_func_evals += res.history[-1].evaluator.n_eval

                # TODO modify a history for NSGA-II-DT
                self.update_history(res, hist_holder, tree_iteration, inner_num_gen, inner_algorithm)

                hist = res.history

                # hist[i] is an object of <class 'pymoo.algorithms.moo.nsga2.NSGA2'>
                best_population_unranked = Population.merge(
                    best_population, res.opt)
                # TODO eliminate duplicates when merging
                best_population = get_nondominated_population(
                    best_population_unranked)
                for generation in hist:
                    all_population = Population.merge(
                        all_population, generation.pop)

            initial_region.population = best_population

            regions = decision_tree.generate_critical_regions(
                all_population, problem, save_folder=None)
            critical_regions = [
                region for region in regions if region.is_critical]
            if not critical_regions:
                critical_regions = [initial_region]
            tree_iteration += 1

        execution_time = time.time() - start_time

        # store the number of achieved tree iterations
        self.done_tree_iterations = tree_iteration
        result = self.create_result(problem, hist_holder, inner_algorithm, execution_time)
        self.res = result
        return result

    def update_history(self, res, hist_holder, tree_iteration, inner_num_gen, inner_algorithm):
        for i in range(inner_num_gen):
            pop = Population.merge(
                hist_holder[tree_iteration * inner_num_gen + i].pop, res.history[i].pop)
            # copy a template of the inner algorithm, and then modify its population and other properties
            algo = copy.deepcopy(inner_algorithm)
            algo.pop = pop
            opt_pop = Population(
                individuals=calc_nondominated_individuals(pop))
            algo.opt = opt_pop
            hist_holder[tree_iteration * inner_num_gen + i] = algo

    def create_result(self, problem, hist_holder, inner_algorithm, execution_time):
        # TODO calculate res.opt
        I = 0
        for algo in hist_holder:
            I += len(algo.pop)
            algo.evaluator.n_eval = I
            algo.start_time = 0
            algo.problem = problem
            algo.result()

        res_holder = ResultExtended()
        res_holder.algorithm = inner_algorithm
        res_holder.algorithm.evaluator.n_eval = I
        res_holder.problem = problem
        res_holder.algorithm.problem = problem
        res_holder.history = hist_holder
        res_holder.exec_time = execution_time

        # calculate total optimal population using individuals from all iterations
        opt_all = Population()
        for algo in hist_holder:
            opt_all = Population.merge(opt_all, algo.pop)
        # log.info(f"opt_all: {opt_all}")
        opt_all_nds = get_nondominated_population(opt_all)
        res_holder.opt = opt_all_nds

        return res_holder
    
    def write_results(self, 
                        ref_point_hv, 
                        ideal, 
                        nadir, 
                        results_folder = RESULTS_FOLDER):

        algorithm_name = self.algorithm_name
        if self.res is None:
            log.info(
                "Result object is None. Execute algorithm first, before writing results.")
            return
        config = self.config
        res = self.res
        res_holder = self.res
        config = self.config
        problem = self.problem
        hist_holder = res_holder.history

        done_tree_iterations = self.done_tree_iterations

        '''For forwarding to plotter'''
        algorithm_parameters = {
            'Number of maximal tree generations': str(config.max_tree_iterations),
            'Number performed tree iterations': done_tree_iterations,
            "Population size": str(config.population_size),
            "Number of generations": str(config.inner_num_gen),
            "Number of offsprings": str(config.num_offsprings),
            "Crossover probability": str(config.prob_crossover),
            "Crossover eta": str(config.eta_crossover),
            "Mutation probability": str(config.prob_mutation),
            "Mutation eta": str(config.eta_mutation)}

        log.info(f"=====[{algorithm_name}] Writing results...")

        save_folder = output.create_save_folder(res.problem, results_folder, algorithm_name, is_experimental=EXPERIMENTAL_MODE)
        
        # output.igd_analysis(res, save_folder)
        # output.gd_analysis(res,save_folder)
     
        output.hypervolume_analysis(res, 
          save_folder, 
          critical_only=True,
          ref_point_hv=ref_point_hv, 
          ideal=ideal, 
          nadir=nadir)        
        # output.spread_analysis(res, save_folder)        # output.spread_analysis(res, save_folder)
        output.write_calculation_properties(res,save_folder,algorithm_name,algorithm_parameters)
        output.design_space(res, save_folder)
        output.objective_space(res, save_folder)
        output.optimal_individuals(res, save_folder)
        output.write_summary_results(res, save_folder)
        output.write_simulation_output(res,save_folder)
        output.simulations(res, save_folder)
        output.all_critical_individuals(res, save_folder)
        output.write_generations(res, save_folder)

        if WRITE_ALL_INDIVIDUALS:
            output.all_individuals(res, save_folder)

        #persist results object
        #res.persist(save_folder + "backup")
        ####################### MNIST SPECIFIC ####################

        # HACK: Write MNIST specific results
        from problem.mnist_problem import MNISTProblem
        if type(self.problem) == MNISTProblem:
            print("Exporting inputs ...")
            # output_mnist.output_optimal_digits(res, save_folder)
            # output_mnist.output_explored_digits(res, save_folder)
            # output_mnist.output_critical_digits(res, save_folder)
            output_mnist.output_critical_digits_all(res, save_folder)
            # output_mnist.output_seed_digits(res, save_folder)
            output_mnist.output_optimal_digits_all(res, save_folder)
            output_mnist.output_seed_digits_all(res, save_folder)
            # output_mnist.output_summary(res, save_folder)
            output_mnist.write_generations_digit(res, save_folder)