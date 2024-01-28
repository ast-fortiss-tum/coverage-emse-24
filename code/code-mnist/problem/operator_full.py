from dataclasses import dataclass
from typing import Dict
from pymoo.core.problem import Problem
import numpy as np
from problem.mnist.archive import Archive
from evaluation.critical import Critical
from evaluation.fitness import *
import logging as log
import sys
import random
from os.path import join
from pathlib import Path
# For Python 3.6 we use the base keras
import keras
from problem.mnist.digit_mutator import DigitMutator
# local imports
from problem.mnist import vectorization_tools
from problem.mnist.digit_input import Digit
from problem.mnist.exploration import Exploration
from model_ga.population import PopulationExtended
from problem.mnist.properties import NGEN, \
    POPSIZE, EXPECTED_LABEL, INITIALPOP, \
    ORIGINAL_SEEDS, BITMAP_THRESHOLD, FEATURES
from problem.mnist import predictor
from math import ceil
import string
from numbers import Real
import random
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
import string
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.crossover import Crossover
from pymoo.core.variable import Real, get
from pymoo.util.misc import row_at_least_once_true
from pymoo.core.crossover import Crossover
from pymoo.core.population import Population
from pymoo.util.misc import crossover_mask
    
from random import randint
from pymoo.operators.crossover.ux import UniformCrossover

import matplotlib.pyplot as plt
import numpy as np

from pymoo.core.individual import Individual
from pymoo.core.problem import Problem
from pymoo.operators.crossover.pntx import PointCrossover, SinglePointCrossover, TwoPointCrossover
import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.util.misc import crossover_mask
import re
import xml.etree.ElementTree as ET
from problem.mnist.digit_input import Digit
from problem.mnist import mutation_manager
from problem.mnist import rasterization_tools
from problem.mnist import vectorization_tools
import random

class CrossoverExample(Crossover):
    def __init__(self):

        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):

        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        # The output owith the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=object)

        print(f"in: {X}\n")
        # for each mating provided
        for k in range(n_matings):

            # get the first and the second parent
            a, b = X[0, k, 0], X[1, k, 0]

            # prepare the offsprings
            off_a = ["_"] * problem.n_characters
            off_b = ["_"] * problem.n_characters

            for i in range(problem.n_characters):
                if np.random.random() < 0.5:
                    off_a[i] = a[i]
                    off_b[i] = b[i]
                else:
                    off_a[i] = b[i]
                    off_b[i] = a[i]

            # join the character list and set the output
            Y[0, k, 0], Y[1, k, 0] = "".join(off_a), "".join(off_b)

        print(f"out: {Y}")
        return Y
    
def mut_binomial(n, m, prob, at_least_once=True):
    prob = np.ones(n) * prob
    M = np.random.random((n, m)) < prob[:, None]

    if at_least_once:
        M = row_at_least_once_true(M)
    return M

class MnistBinomialCrossover(Crossover):

    def __init__(self, bias=0.5, n_offsprings=2, **kwargs):
        super().__init__(2, n_offsprings, **kwargs)
        self.bias = Real(bias, bounds=(0.1, 0.9), strict=(0.0, 1.0))

        log.info("[MnistBinomialCrossover] Custom crossover init")

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape

        # log.info(f"[Crossover] shape: {X.shape}")

        Xp = np.full((self.n_offsprings, n_matings, problem.n_var), None, dtype=object)
        
        bias = get(self.bias, size=n_matings)
        M = mut_binomial(n_matings, n_var, bias, at_least_once=True)
        # print(f"[Crossover] M is {M}")
        
        if self.n_offsprings == 1:
            Xp = X[0].copy(X)
            Xp[~M] = X[1][~M]
        elif self.n_offsprings == 2:
            Xp = np.copy(X)
            Xp[0][~M] = X[1][~M]
            Xp[1][~M] = X[0][~M]
        else:
            raise Exception
        
        # print(f"[Crossover] Result {Xp}")
        
        # # Do nothing
        
        # Xp = X

        # log.info(f"[Crossover] output shape: {Xp.shape}")
        return Xp

class UniformCrossover(Crossover):

    def __init__(self, **kwargs):
        super().__init__(2, 2, **kwargs)

    def _do(self, _, X, **kwargs):
        _, n_matings, n_var = X.shape
        M = np.random.random((n_matings, n_var)) < 0.5
        _X = crossover_mask(X, M)
        return _X


class UX(UniformCrossover):
    pass

class MyNoCrossover(Crossover):
    def __init__(self):
        super().__init__(1, 1, 0.0)

    def do(self, problem, pop, **kwargs):
        res = Population.create(*[np.random.choice(parents) for parents in pop])
        return res

class MnistMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        # print(f"[Mutation] received {X}")

        Y= np.full((len(X), 1), None, dtype=object)
        # for each individual
        for i in range(len(X)):
            d_old = X[i][0]
            new_digit = d_old.clone()
            DigitMutator(new_digit).mutate()
            Y[i,0] = new_digit
        # assert d_old.xml_desc != new_digit.xml_desc
        return Y
        
class MyCrossover(Crossover):
    def __init__(self):

        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):

        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        # The output owith the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=object)

        # print(f"in: {X}\n")
        # for each mating provided
        for k in range(n_matings):

            # # get the first and the second parent
            digit_1, digit_2 = X[0, k, 0], X[1, k, 0]

            if random.random() < 0.5:
                off_1, off_2 = exchange_cp(digit_1, digit_2)
                Y[0, k, 0], Y[1, k, 0] = off_1, off_2

            else:
                Y[0, k, 0], Y[1, k, 0] = digit_1, digit_2

        # print(f"out: {Y}")
        return Y

def exchange_cp(digit_1, digit_2, ratio=0.1):
    # TODO exchange a randomly chosen control point
    off_1 = digit_1.clone()
    off_2 = digit_2.clone()

    segments_1, path_1 = parse_svg(off_1.xml_desc)
    segments_2, path_2 = parse_svg(off_2.xml_desc)

    num_matches_1 = len(segments_1)
    num_matches_2 = len(segments_2)

    # print(segments_1)
    # print(segments_2)

    if num_matches_1 > 2:
        random_coordinate_index = randint(0, num_matches_1 - 1)
        # focus on one              
        segment1 = segments_1[random_coordinate_index]
        old_cp_1 = "C " + segment1[0] + ',' + segment1[1] + ' ' + segment1[2] + ',' + segment1[3] + ' ' + segment1[4] + ',' + segment1[5] + ' '
        
        random_coordinate_index_other = randint(0, num_matches_2 - 1)
        segment2 = segments_2[random_coordinate_index_other]
        old_cp_2 = "C " + segment2[0] + ',' + segment2[1] + ' ' + segment2[2] + ',' + segment2[3] + ' ' + segment2[4] + ',' + segment2[5] + ' '
        
        def get_new_point(cp1, cp2):
            return ratio*(float(cp2)-float(cp1)) + float(cp1) 

        # mate
        new_cp_1 = "C " + str(get_new_point(segment1[0], segment2[0])) + ',' + str(get_new_point(segment1[1], segment2[1]))   + ' ' + segment1[2] + ',' + segment1[3] + ' ' + segment1[4] + ',' + segment1[5] + ' '
        new_cp_2 = "C " + str(get_new_point(segment2[0], segment1[0])) + ',' + str(get_new_point(segment2[1], segment1[1]))  + ' ' + segment2[2] + ',' + segment2[3] + ' ' + segment2[4] + ',' + segment2[5] + ' '

        path_new_1 = re.sub(old_cp_1, new_cp_1, path_1)
        path_new_2 = re.sub(old_cp_2, new_cp_2, path_2)
        off_1 = vectorization_tools.create_svg_xml(path_new_1)
        off_2 = vectorization_tools.create_svg_xml(path_new_2)
        digit_off_1 = Digit(off_1, digit_1.expected_label, digit_1.seed)
        digit_off_2 = Digit(off_2, digit_2.expected_label, digit_2.seed)
    else:
        off_1 = digit_1
        off_2 = digit_2

    print("Checkpoints exchanged.")
    return digit_off_1, digit_off_2

   
def parse_svg(svg_desc):
    NAMESPACE = '{http://www.w3.org/2000/svg}'

    root = ET.fromstring(svg_desc)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    # find all the vertexes
    pattern = re.compile('C\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
    segments = pattern.findall(svg_path)
    # chose a random control point
    return segments, svg_path
