import pymoo

from algorithm.ps import PureSampling
from experiment.search_configuration import SearchConfiguration
from model_ga.problem import ProblemExtended
from utils.fps import FPS
pymoo.core.problem.Problem = ProblemExtended
from pymoo.core.problem import Problem
from utils.sampling import CartesianSampling

import os

class PureSamplingFPS(PureSampling):
    def __init__(self,
                    problem: Problem,
                    config: SearchConfiguration,
                    sampling_type = FPS):
        super().__init__(
            problem = problem,
            config = config,
            sampling_type = sampling_type)
        
