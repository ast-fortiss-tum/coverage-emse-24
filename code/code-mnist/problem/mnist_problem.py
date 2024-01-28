from dataclasses import dataclass
from math import ceil
from typing import Dict
from pymoo.core.problem import Problem
from pymoo.core.individual import Individual
import numpy as np
from problem.mnist.archive import Archive
from evaluation.critical import Critical
from evaluation.fitness import *
import logging as log
import sys
import time
import traceback
import random
from os.path import join
from pathlib import Path
# For Python 3.6 we use the base keras
import keras
from problem.mnist.digit_mutator import DigitMutator
# local imports
from problem.mnist import vectorization_tools
from problem.mnist.digit_input import Digit
from model_ga.population import PopulationExtended
from problem.mnist.properties import NGEN, \
    POPSIZE, EXPECTED_LABEL, INITIALPOP, \
    ORIGINAL_SEEDS, BITMAP_THRESHOLD, FEATURES
from problem.mnist import predictor
from utils.math_utils import round
from problem.mnist import features
from scipy.stats import entropy
from problem.mnist import utils_mnist
from utils import string_utils
from problem.mnist_loader import mnist_loader

@dataclass
class MNISTProblem(Problem):

    def __init__(self,
                 xl: List[float], 
                 xu: List[float], 
                 fitness_function: Fitness, 
                 critical_function: Critical, 
                 simulation_variables: List[float], 
                 design_names: List[str] = None, 
                 objective_names: List[str] = None, 
                 problem_name: str = None, 
                 other_parameters: Dict = None,
                 expected_label: int = None,
                 max_seed_distance: float = 3,
                 min_saturation: float = 0,
                 seed: int = None):

        super().__init__(n_var=len(xl),
                         n_obj=len(fitness_function.name),
                         xl=xl,
                         xu=xu)

        assert xl is not None
        assert xu is not None
        assert np.equal(len(xl), len(xu))
        assert np.less_equal(xl, xu).all()
        assert expected_label is not None

        self.set_fitness_function(fitness_function, objective_names)

        self.critical_function = critical_function
        self.simulation_variables = simulation_variables
        self.expected_label = expected_label
        self.max_seed_distance = max_seed_distance
        self.min_saturation = min_saturation
        self.problem_name = problem_name

        print(f"problem config: {self.__dict__}")

        self.set_seed(seed)

        if design_names is not None:
            self.design_names = design_names
        else:
            self.design_names = simulation_variables


        if other_parameters is not None:
            self.other_parameters = other_parameters

        self.counter = 0

    def set_seed(self, seed):
        log.info(f"Seed of MNISTProblem set to: {seed}")

        digit = MNISTProblem.generate_and_evaluate_digit(seed)
        segment_num = utils_mnist.get_number_segments(digit)
        vertex_num = utils_mnist.get_number_verts(digit)

      
        
        # adapt the search space as number of  vertices is required in space specification
        s_size = len(self.xu)
        if s_size == 3:
            self.xu[s_size-1] = vertex_num - 1
        if s_size == 8:
            self.xu[s_size-1] = vertex_num - 1
            self.xu[s_size-2] = vertex_num - 1

        self.seed = seed
        self.vertex_num = vertex_num
        self.segment_num = segment_num
        self.seed_digits = [
               digit
        ]
        # hack: replace name so that seed is in name
        self.problem_name = string_utils.replace_last_number_with_new_number(self.problem_name, seed)
                
    def set_fitness_function(self, fitness_function, objective_names = None):
        assert fitness_function is not None
        assert len(fitness_function.min_or_max) == len(fitness_function.name)
        
        self.n_obj=len(fitness_function.name)

        self.fitness_function = fitness_function

        if objective_names is not None:
            self.objective_names = objective_names
        else:
            self.objective_names = fitness_function.name

        self.signs = []
        for value in self.fitness_function.min_or_max:
            if value == 'max':
                self.signs.append(-1)
            elif value == 'min':
                self.signs.append(1)
            else:
                raise ValueError(
                    "Error: The optimization property " + str(value) + " is not supported.")

        
    def _evaluate(self, x, out, *args, **kwargs):
        archive = kwargs.get("archive")
        if archive is not None:
            print(f"received archive length: {len(archive)}")
        else:
            #create empty archive # TODO should be managed by Algorithm
            archive = PopulationExtended()

        vector_list = []
        label_list = []
        digits = []
        self.counter = self.counter + 1

        for i, ind in enumerate(x):

            # # print(f"input: {ind}")
            # extent_1 = ind[0]
            # extent_2 = ind[1] 

            # extent_3 = ind[2]
            # extent_4 = ind[3]
            # c_index_1 = ind[4]
            # c_index_2 = ind[5]
            # # apply mutations only to the same seed digit
            # new_digit = self.seed_digits[0].clone()
    
            # # Perform displacement on first control point
            # vertex_1 =  round(c_index_1)
            # vertex_2 =  round(c_index_2)

            ########################################

            extent_1 = ind[0]
            extent_2 = ind[1] 
            vertex = round(ind[2])

            # apply mutations only to the same seed digit
            new_digit = self.seed_digits[0].clone()
    
            ########## COORDINATE WISE MUTATION ##############

            new_digit = self.apply_mutation_index(new_digit, extent_1, extent_2, vertex)
            
            # new_digit = self.apply_mutation_index_bi(new_digit, 
            #                 extent_1, 
            #                 extent_2, 
            #                 extent_3, 
            #                 extent_4, 
            #                 vertex_1, 
            #                 vertex_2)

            ########## SEGMENT WISE MUTATION ##############
            # new_digit = self.apply_mutation_segment(self, new_digit, extent_1, extent_2, vertex)

            ####################################
            # # Remove a point
            #DigitMutator(new_digit).mutate(extent_2, c_index = ((round(c_index) + 3) % self.vertex_num), mutation=3)
            assert(new_digit.seed == self.seed_digits[0].seed)
            
            ##### Evalute fitness value of the classification ( = simulation) ########
            predicted_label, confidence = \
                    predictor.Predictor.predict(new_digit.purified)
            predictions = predictor.Predictor.predict_extended(new_digit.purified)

            ##### store info in digit ##########
            new_digit.predicted_label = predicted_label
            new_digit.confidence = confidence

            # distance = self.archive.get_min_distance_from_archive(new_digit)
            distance = get_min_distance_from_archive(digit=new_digit, archive=archive)
            distance_seed = new_digit.distance(self.seed_digits[0])
            distance_test_input = get_min_distance_from_archive_input(ind, archive=archive)

            brightness = new_digit.brightness(min_saturation=self.min_saturation)
            coverage = new_digit.coverage(min_saturation=self.min_saturation)
            coverage_rel = new_digit.coverage(
                                        min_saturation=self.min_saturation,
                                        relative = True
            )
            # print(f"new digit has coverage: {coverage}")
            # print(f"new digit has distance (seed): {distance_seed}")
            # print(f"new digit has distance (archive)): {distance}")

            data = {}
            data["predicted_label"] = predicted_label
            data["confidence"] = confidence
            data["predictions"] = predictions
            data["expected_label"] = self.expected_label
            data["archive"] = archive # all digits found so far # TODO improve how we pass the archive
            data["digit"] = new_digit
            data["distance_archive"] = distance
            data["coverage"] = coverage
            data["brightness"] = brightness
            data["move_distance"] = features.move_distance(new_digit)
            data["angle"] = features.angle_calc(new_digit)
            data["orientation"] = features.orientation_calc(new_digit, self.min_saturation)
            data["entropy_signed"] = -entropy(pk=predictions) if np.argmax(predictions) != self.expected_label else entropy(pk=predictions)
            data["distance_test_input"] = distance_test_input
            data["coverage_rel"] = coverage_rel
            
            vector_fitness = np.asarray(
                self.fitness_function.eval(simout=None,data=data)
            )
            # set structure
            signed_fitness = np.asarray(self.signs) * np.array(vector_fitness)
            vector_list.append(signed_fitness)
            label_list.append(self.critical_function.eval(signed_fitness,simout=None))
            digits.append(new_digit)

        out["F"] = np.vstack(vector_list)
        out["CB"] = label_list
        out["DIG"] = digits
        
        log.info("Individual evaluated and mutated digit created.")

    def is_simulation(self):
        return False

    def apply_mutation_index_bi(self, new_digit, extent_1, extent_2,extent_3, extent_4, index_1, index_2):  
        # assure that x,y coordinage of the same point are mutated
        if index_1 % 2 == 0:
            next_vertex = index_1 + 1
        else:
            next_vertex = index_1- 1
        # next_vertex = index + 1

        DigitMutator(new_digit).mutate(extent_1, c_index = index_1, mutation=2)
        DigitMutator(new_digit).mutate(extent_2, c_index = (next_vertex % self.vertex_num), mutation=2)
        
        # DigitMutator(new_digit).mutate(extent_1, c_index = index + 2, mutation=2)
        # DigitMutator(new_digit).mutate(extent_2, c_index = (next_vertex + 2 % self.vertex_num), mutation=2)
        if index_2 % 2 == 0:
            next_vertex = index_2 + 1
        else:
            next_vertex = index_2 - 1
        # Perform displacement on inner point
        DigitMutator(new_digit).mutate(extent_3, c_index = index_2, mutation=1)
        DigitMutator(new_digit).mutate(extent_4, c_index = (next_vertex % self.vertex_num), mutation=1)
        
        # Perform displacement on inner point
        # DigitMutator(new_digit).mutate(extent_1, c_index = (index + 2 ) % self.vertex_num, mutation=1)
        # DigitMutator(new_digit).mutate(extent_2, c_index = ((next_vertex + 2) % self.vertex_num), mutation=1)

        return new_digit

    def apply_mutation_index(self, new_digit, extent_1, extent_2, index):  
        # assure that x,y coordinage of the same point are mutated
        if index % 2 == 0:
            next_vertex = index + 1
        else:
            next_vertex = index - 1
        # next_vertex = index + 1

        DigitMutator(new_digit).mutate(extent_1, c_index = index, mutation=2)
        DigitMutator(new_digit).mutate(extent_2, c_index = (next_vertex % self.vertex_num), mutation=2)
        
        # DigitMutator(new_digit).mutate(extent_1, c_index = index + 2, mutation=2)
        # DigitMutator(new_digit).mutate(extent_2, c_index = (next_vertex + 2 % self.vertex_num), mutation=2)
        
        # Perform displacement on inner point
        DigitMutator(new_digit).mutate(extent_1, c_index = index, mutation=1)
        DigitMutator(new_digit).mutate(extent_2, c_index = (next_vertex % self.vertex_num), mutation=1)
        
        # Perform displacement on inner point
        DigitMutator(new_digit).mutate(extent_1, c_index = (index + 2 ) % self.vertex_num, mutation=1)
        DigitMutator(new_digit).mutate(extent_2, c_index = ((next_vertex + 2) % self.vertex_num), mutation=1)

        return new_digit


    def apply_mutation_segment(self, new_digit, extent_1, extent_2, vertex):              
        DigitMutator(new_digit).mutate_point(extent_x=extent_1, 
                                             extent_y=extent_2, 
                                             segment=vertex, 
                                             mutation=1)
                                                
        DigitMutator(new_digit).mutate_point(extent_x= 0.5* extent_1, 
                                            extent_y= 0.5 * extent_2, 
                                            segment=(vertex + 3) % self.segment_num, 
                                            mutation=2)

        DigitMutator(new_digit).mutate_point(extent_x=0.5* extent_1, 
                                extent_y=0.5* extent_2, 
                                segment=(vertex + 5) % self.segment_num, 
                                mutation=2)
        return new_digit

    def generate_digit(seed):
        seed_image =  mnist_loader.get_x_test()[int(seed)]
        xml_desc = vectorization_tools.vectorize(seed_image)
        return Digit(xml_desc, EXPECTED_LABEL, seed)
    
    # get predicitons and metrics for digits for input validation
    def generate_and_evaluate_digit(seed):
        seed_image = mnist_loader.get_x_test()[int(seed)]
        xml_desc = vectorization_tools.vectorize(seed_image)

        digit =  Digit(xml_desc, EXPECTED_LABEL, seed)

        predicted_label, confidence = predictor.Predictor.predict(digit.purified)

        digit.confidence = confidence
        digit.predicted_label = predicted_label

        return digit

def get_class_from_function(f):
    return vars(sys.modules[f.__module__])[f.__qualname__.split('.')[0]] 


def get_min_distance_from_archive(digit: Digit, archive: PopulationExtended):
    distances = []
    for archived_digit in archive.get("DIG"):
        # print("Digit or some close digit is in archive.")
        if archived_digit.purified is not digit.purified:
            dist = np.linalg.norm(archived_digit.purified - digit.purified)
            # TODO fix, distance is somehow 0, even when digit not in archive
            # if dist == 0:
            #     print("Distance is 0, skip.")
            #     continue
            distances.append(dist)
    if len(distances) == 0:
        return 0
    else:
        min_dist = min(distances)
    return min_dist

def get_min_distance_from_archive_input(X_ind, archive: PopulationExtended):
    distances = []
    for X in archive.get("X"):
        # print("Digit or some close digit is in archive.")
        if X_ind is not X:
            dist = np.linalg.norm(X - X_ind)
            distances.append(dist)
    if len(distances) == 0:
        return 0
    else:
        min_dist = min(distances)
    return min_dist