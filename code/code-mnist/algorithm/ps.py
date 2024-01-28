

import pymoo
from simulation.simulator import SimulationOutput
from utils.sampling import CartesianSampling

# from model_ga.individual import IndividualSimulated
# pymoo.core.individual.Individual = IndividualSimulated

# from model_ga.population import PopulationExtended
# pymoo.core.population.Population = PopulationExtended

# from model_ga.result  import ResultExtended
# pymoo.core.result.Result = ResultExtended

# from model_ga.problem import ProblemExtended
# pymoo.core.problem.Problem = ProblemExtended

from pymoo.core.algorithm import Algorithm

import os
import sys
from pathlib import Path
from typing import List

from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from algorithm.classification.classifier import ClassificationType
from algorithm.classification.decision_tree.decision_tree import *
from evaluation.critical import Critical
from experiment.search_configuration import DefaultSearchConfiguration, SearchConfiguration
from problem.pymoo_test_problem import PymooTestProblem
from simulation.simulator import SimulationOutput
from visualization import output
import quality_indicators.metrics.spread as qi
import logging as log
from utils.evaluation import evaluate_individuals
from model_ga.result import *
import time
from visualization import output_mnist
from config import *

class PureSampling(object):
    
    algorithm_name = "RS"

    def __init__(self,
                problem: Problem,
                config: SearchConfiguration,
                sampling_type = FloatRandomSampling):

        self.config = config
        self.problem = problem
        self.res = None
        self.sampling_type = sampling_type
        self.sample_size = config.population_size
        self.n_splits = 10 # divide the population by this size 
                         # to make the algorithm iterative for further analysis
        log.info(f"Initialized algorithm with config: {config.__dict__}")

    def run(self) -> ResultExtended:

        problem = self.problem
        sample_size = self.sample_size
        sampled = self.sampling_type()(problem,sample_size)
        n_splits = self.n_splits
        start_time = time.time()

        pop = evaluate_individuals(sampled, problem)

        execution_time = time.time() - start_time

        # create result object

        self.res = PureSampling.create_result(problem, pop, execution_time, n_splits)
        
        return self.res 
    
    @staticmethod
    def create_result(problem, pop, execution_time, n_splits):
        res_holder = ResultExtended()
        res_holder.algorithm = Algorithm()
        res_holder.algorithm.pop = pop
        res_holder.algorithm.evaluator.n_eval = len(pop)
        res_holder.problem = problem
        res_holder.algorithm.problem = problem
        res_holder.exec_time = execution_time
        res_holder.opt = get_nondominated_population(pop)
        res_holder.algorithm.opt = res_holder.opt

        res_holder.history = []  # history is the same instance 
        n_bucket = len(pop) // n_splits
        
        pop_sofar = 0
        for i in range(0,n_splits):
            
            algo = Algorithm()
            algo.pop = pop[(i*n_bucket):min((i+1)*n_bucket,len(pop))]
            pop_sofar += len(algo.pop)
            algo.evaluator.n_eval = pop_sofar
            algo.opt = get_nondominated_population(algo.pop)
            res_holder.history.append(algo)
        
        return res_holder

    def write_results(self, 
                        ref_point_hv, 
                        ideal, 
                        nadir, 
                        results_folder = RESULTS_FOLDER):

        algorithm_name = self.algorithm_name
        if self.res is None:
            log.info("Result object is None. Execute algorithm first, before writing results.")
            return
        log.info(f"=====[{self.algorithm_name}] Writing results...")
        config = self.config
        res = self.res
        algorithm_parameters = {
            "Number of samples" : str(self.sample_size),
        }
        
        save_folder = output.create_save_folder(res.problem, results_folder, algorithm_name,is_experimental=EXPERIMENTAL_MODE)
        
        # Analysis
        output.igd_analysis(res, save_folder)
        output.gd_analysis(res,save_folder)
        output.hypervolume_analysis(res, 
          save_folder, 
          critical_only=True,
          ref_point_hv=ref_point_hv, 
          ideal=ideal, 
          nadir=nadir) 
          
        output.spread_analysis(res, save_folder)

        # Basis Output
        output.write_calculation_properties(res,save_folder,algorithm_name,algorithm_parameters)
        output.design_space(res, save_folder)
        output.objective_space(res, save_folder)
        output.optimal_individuals(res, save_folder)
        output.write_summary_results(res, save_folder)
        output.write_simulation_output(res,save_folder)
        output.simulations(res, save_folder)
        output.all_critical_individuals(res, save_folder)

        if WRITE_ALL_INDIVIDUALS:
            output.all_individuals(res, save_folder)

        #persist results object
        res.persist(save_folder + "backup")
        
        ####################### MNIST SPECIFIC ####################

        # HACK: Write MNIST specific results
        from problem.mnist_problem import MNISTProblem
        if type(self.problem) == MNISTProblem:
            print("Exporting inputs ...")
            output_mnist.output_optimal_digits(res, save_folder)
            # output_mnist.output_explored_digits(res, save_folder)
            output_mnist.output_critical_digits(res, save_folder)
            output_mnist.output_critical_digits_all(res, save_folder)
            # output_mnist.output_seed_digits(res, save_folder)
            output_mnist.output_optimal_digits_all(res, save_folder)
            output_mnist.output_seed_digits_all(res, save_folder)

