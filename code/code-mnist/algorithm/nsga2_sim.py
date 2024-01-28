
# from model_ga.individual import IndividualSimulated
# pymoo.core.individual.Individual = IndividualSimulated

# from model_ga.population import PopulationExtended
# pymoo.core.population.Population = PopulationExtended

# from model_ga.result  import ResultExtended
# pymoo.core.result.Result = ResultExtended

# from model_ga.problem import ProblemExtended
# pymoo.core.problem.Problem = ProblemExtended

from config import SHOW_PLOT,PLOT_ONLY_LAST_ITERATION
from simulation.simulator import SimulationOutput

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
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from algorithm.classification.classifier import ClassificationType
from algorithm.classification.decision_tree.decision_tree import *
from evaluation.critical import Critical
from experiment.search_configuration import DefaultSearchConfiguration, SearchConfiguration
from problem.pymoo_test_problem import PymooTestProblem
from simulation.simulator import SimulationOutput
from utils.analysis_utils import read_critical_set
from visualization import output
import quality_indicators.metrics.spread as qi
import logging as log
from problem import *
from model_ga.result import *
from visualization import output_mnist
from config import *
from pymoo.operators.sampling.lhs import LHS

class NSGAII_SIM(object):
    
    algorithm_name =  "NSGA-II"

    def __init__(self,
                problem: Problem,
                config: SearchConfiguration):

        self.config = config
        self.problem = problem
        self.res = None

        if self.config.prob_mutation is None:
            self.config.prob_mutation = 1 / problem.n_var

        if config.operators is not None and config.operators["init"]:
            sampling = config.operators["init"]()
        else:
            sampling = LHS()
        
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

        self.algorithm = NSGA2(
            pop_size=config.population_size,
            n_offsprings=config.num_offsprings,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=eliminate_duplicates)

        ''' Prioritize max search time over set maximal number of generations'''
        if config.maximal_execution_time is not None:
            self.termination = get_termination("time", config.maximal_execution_time)
        else:
            self.termination = get_termination("n_gen", config.n_generations)

        self.save_history = True
        
        log.info(f"Initialized algorithm with config: {config.__dict__}")

    def run(self) -> ResultExtended:
        self.res = minimize(self.problem,
                    self.algorithm,
                    self.termination,
                    save_history=self.save_history,
                    verbose=True)

        return self.res

 
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
            "Population size" : str(config.population_size),
            "Number of generations" : str(config.n_generations),
            "Number of offsprings": str(config.num_offsprings),
            "Crossover probability" : str(config.prob_crossover),
            "Crossover eta" : str(config.eta_crossover),
            "Mutation probability" : str(config.prob_mutation),
            "Mutation eta" : str(config.eta_mutation)
        }
        
        save_folder = output.create_save_folder(res.problem, results_folder, algorithm_name, is_experimental=EXPERIMENTAL_MODE)
        
        # output.igd_analysis(res, save_folder)
        # output.gd_analysis(res,save_folder)
     
        output.hypervolume_analysis(res, 
          save_folder, 
          critical_only=True,
          ref_point_hv=ref_point_hv, 
          ideal=ideal, 
          nadir=nadir)    

        # coverage 

        cs_estimated  = read_critical_set(PATH_CRITICAL_SET)
        output.cid_analysis_hitherto(
            res, reference_set=cs_estimated, save_folder=save_folder)
    
        # output.spread_analysis(res, save_folder)        # output.spread_analysis(res, save_folder)
        output.write_calculation_properties(res,save_folder,algorithm_name,algorithm_parameters)
        output.design_space(res, save_folder)
        output.objective_space(res, save_folder, last_iteration=PLOT_ONLY_LAST_ITERATION, show=SHOW_PLOT)
        output.optimal_individuals(res, save_folder)
        output.write_summary_results(res, save_folder)
        output.write_simulation_output(res,save_folder)
        output.simulations(res, save_folder)
        output.all_critical_individuals(res, save_folder)
        output.write_generations(res, save_folder)

        if WRITE_ALL_INDIVIDUALS:
            output.all_individuals(res, save_folder)

        #persist results object
        res.persist(save_folder + "backup")
        res.problem.persist(save_folder + "backup")

        ####################### MNIST SPECIFIC ####################

        # HACK: Write MNIST specific results
        from problem.mnist_problem import MNISTProblem
        if type(self.problem) == MNISTProblem:
            print("Exporting inputs ...")
            # output_mnist.output_optimal_digits(res, save_folder)
            # output_mnist.output_explored_digits(res, save_folder)
            # output_mnist.output_critical_digits(res, save_folder)
            output_mnist.output_critical_digits_all(res, save_folder)
            output_mnist.output_seed_digits(res, save_folder)
            output_mnist.output_optimal_digits_all(res, save_folder)
            output_mnist.output_seed_digits_all(res, save_folder)
            # output_mnist.output_summary(res, save_folder)
            output_mnist.write_generations_digit(res, save_folder)


if __name__ == "__main__":        
    
    # problem = PymooTestProblem(
    #     'BNH', critical_function=CriticalBnhDivided())
    # config = DefaultSearchConfiguration()

    class CriticalMW1(Critical):
        def eval(self, vector_fitness: List[float], simout: SimulationOutput = None):
            if vector_fitness[0] <= 0.8 and vector_fitness[0] >= 0.2 and \
               vector_fitness[1] <= 0.8 and vector_fitness[0] >= 0.2:
                return True
            else:
                return False
            
    problem = PymooTestProblem(
        'mw1', critical_function=CriticalMW1())
    config = DefaultSearchConfiguration()

    config.population_size = 50
    config.inner_num_gen = 5
    config.prob_mutation = 0.5
    config.n_func_evals_lim = 1000
    
    optimizer = NSGAII_SIM(problem,config)
    optimizer.run()
    optimizer.write_results()