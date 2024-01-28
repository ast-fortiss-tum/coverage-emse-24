import os
from evaluation.fitness import *
from evaluation import critical

from problem.adas_problem import ADASProblem
from problem.pymoo_test_problem import PymooTestProblem
from experiment.experiment import *
from algorithm.algorithm import *
from evaluation.critical import *
from problem.mnist_problem import *
from problem.fitness_mnist import *
from problem.mnist.utils_mnist import get_number_segments, get_number_verts
from problem.operator import MnistSamplingValid
import copy
from config import *

''' MNIST Problem with single seed'''
config = DefaultSearchConfiguration()
config.population_size = 20
config.n_generations =  50
config.operators["init"] = MnistSamplingValid

seed = 120 #127 #52# 132 #129
#other possible seeds: 8, 15, 23, 45, 52, 53, 102, 120, 127, 129, 132, 152
lb = -8
ub = +8

digit = MNISTProblem.generate_digit(seed)
vertex_num = get_number_verts(digit)
ub_vert = vertex_num -1 

# config.operators["mut"] = MnistMutation
# config.operators["cx"] = MyNoCrossover
# config.operators["dup"] = MnistDuplicateElimination
config.operators["init"] = MnistSamplingValid

mnistproblem = MNISTProblem(
                        problem_name=f"MNIST_6D",
                        xl=[lb, lb, lb, lb,  0, 0],
                        xu=[ub, ub, ub, ub,  ub_vert, ub_vert],
                        simulation_variables=[
                            "mut_extent_1",
                            "mut_extent_2",
                            "mut_extent_3",
                            "mut_extent_4",
                            "vertex_control",
                            "vertex_start"
                        ],
                        fitness_function=FitnessMNIST(),
                        critical_function=CriticalMNISTConf_05(),
                        expected_label=5,
                        min_saturation=0.1,
                        max_seed_distance=4,
                        seed=seed
                        )

mnistproblem = MNISTProblem(
                        problem_name=f"MNIST_3D",
                        xl=[lb, lb, 0],
                        xu=[ub, ub, ub_vert],
                        simulation_variables=[
                            "mut_extent_1",
                            "mut_extent_2",
                            "vertex_control"
                        ],
                        fitness_function=FitnessMNIST(),
                        critical_function=CriticalMNISTConf_05(),
                        expected_label=5,
                        min_saturation=0.1,
                        seed=seed
                        )
#############################################

''' NSGA-II with optimizing diversity'''
def getExp5() -> Experiment:
    problem = copy.deepcopy(mnistproblem)
    problem.set_fitness_function(FitnessMNIST(diversify=True))
    problem.critical_function=CriticalMNISTConf_05()
    problem.problem_name = problem.problem_name+ "_NSGA-II-D" + f"_D{seed}" 
    experiment = Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)
    return experiment

''' Plain NSGA-II'''
def getExp6() -> Experiment:
    problem = copy.deepcopy(mnistproblem)
    problem.set_fitness_function(FitnessMNIST(diversify=False))
    problem.critical_function=CriticalMNISTConf_05()
    problem.problem_name = problem.problem_name + "_NSGA-II" + f"_D{seed}" 
    experiment = Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)
    return experiment
''' Grid sampling '''
def getExp7() -> Experiment:
    problem = copy.deepcopy(mnistproblem)
    problem.set_fitness_function(FitnessMNIST(diversify=False))
    problem.critical_function=CriticalMNISTConf_05()
    problem.problem_name = problem.problem_name + "_GS" + f"_D{seed}" 
    experiment = Experiment(problem=problem,
                            algorithm=AlgorithmType.PS_GRID,
                            search_configuration=config)
    return experiment

''' NSGA-II-DT '''
def getExp8() -> Experiment:
    problem = copy.deepcopy(mnistproblem)
    problem.set_fitness_function(FitnessMNIST(diversify=False))
    problem.critical_function=CriticalMNISTConf_05()
    problem.problem_name = problem.problem_name + "_NSGA-II-DT" +  f"_D{seed}" 
    config.inner_num_gen = 5
    experiment = Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAIIDT,
                            search_configuration=config)
    return experiment

''' RS '''
def getExp9() -> Experiment:
    print("running exp 9")
    config = DefaultSearchConfiguration()
    config.operators["init"] = MnistSamplingValid
    config.population_size = 1000
    print(f"using config: {config.__dict__}")
    problem = copy.deepcopy(mnistproblem)
    problem.set_fitness_function(FitnessMNIST(diversify=True))
    problem.critical_function=CriticalMNISTConf_05()
    problem.problem_name = problem.problem_name + "_RS" +  f"_D{seed}" 
    experiment = Experiment(problem=problem,
                            algorithm=AlgorithmType.PS_RAND,
                            search_configuration=config)
    return experiment

''' GRID '''
def getExp10() -> Experiment:
    print("running exp 10")
    print(f"using config: {config.__dict__}")
    problem = copy.deepcopy(mnistproblem)
    problem.set_fitness_function(FitnessMNIST(diversify=False))
    problem.critical_function=CriticalMNISTConf_05()
    problem.problem_name = problem.problem_name + "_GS" +  f"_D{seed}" 
    config.population_size = 10

    experiment = Experiment(problem=problem,
                            algorithm=AlgorithmType.PS_GRID,
                            search_configuration=config)
    return experiment

############### DummySimulator Problems #######################

from simulation.dummy_simulation import DummySimulator

dummy_problem = ADASProblem(
                          problem_name="DummySimulatorProblem",
                          scenario_path="",
                          xl=[0, 1, 0, 1],
                          xu=[360, 3,360, 3],
                          simulation_variables=[
                              "orientation_ego",
                              "velocity_ego",
                              "orientation_ped",
                              "velocity_ped"],
                          fitness_function=FitnessAdaptedDistanceSpeed(),
                          critical_function=CriticalAdasAdaptedDistanceVelocity(),
                          simulate_function=DummySimulator.simulate,
                          simulation_time=10,
                          sampling_time=0.25
                          )

def getExp100() -> Experiment:
    config = DefaultSearchConfiguration()
    config.population_size = 10
    config.n_generations = 10
    config.operators["init"] = None

    config.ideal = np.asarray([-1,-5])
    config.nadir = np.asarray([-0.79,-0.09])

    experiment = Experiment(problem=dummy_problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)
    return experiment

def getExp101() -> Experiment:
    config = DefaultSearchConfiguration()
    config.maximal_execution_time = None
    config.inner_num_gen = 5
    config.population_size = 10
    config.n_func_evals_lim = 100
    config.n_generations = None
    config.operators["init"] = None

    config.ideal = np.asarray([-1,-5])
    config.nadir = np.asarray([-0.79,-0.09])

    # config.maximal_execution_time = None # limit the total time, to control the number of tree iterations
    experiment = Experiment(problem=dummy_problem,
                            algorithm=AlgorithmType.NSGAIIDT,
                            search_configuration=config)
    return experiment

experiment_switcher = {
    5: getExp5,
    6: getExp6,
    7: getExp7,
    8: getExp8,
    9: getExp9,
    10: getExp10,
    100: getExp100,
    101: getExp101
    }
