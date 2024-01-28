import pymoo

from algorithm.ps import PureSampling
from experiment.search_configuration import SearchConfiguration
from model_ga.problem import ProblemExtended
pymoo.core.problem.Problem = ProblemExtended
from pymoo.core.problem import Problem
from utils.sampling import CartesianSampling

import os
class PureSamplingGrid(PureSampling):
    def __init__(self,
                    problem: Problem,
                    config: SearchConfiguration,
                    sampling_type = CartesianSampling):
        super().__init__(
            problem = problem,
            config = config,
            sampling_type = sampling_type)
