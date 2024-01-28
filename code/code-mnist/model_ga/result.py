import numpy as np
from utils.sorting import *
from pymoo.core.result import Result
from model_ga.population import PopulationExtended as Population
from model_ga.individual import IndividualSimulated as Individual
import dill
import os
from pathlib import Path

class ResultExtended(Result):

    def __init__(self) -> None:
        super().__init__()
    
    def obtain_history_design(self):
        hist = self.history
        
        if hist is not None:
            n_evals = []  # corresponding number of function evaluations
            hist_X = []  # the objective space values in each 
            pop = Population()
            for algo in hist:
                n_evals.append(algo.evaluator.n_eval)  # store the number of function evaluations                            
                pop = Population.merge(pop, algo.pop)
                feas = np.where(pop.get("feasible"))[
                    0]  # filter out only the feasible and append and objective space values
                hist_X.append(pop.get("X")[feas])
        else:
            n_evals = None
            hist_X = None
        return n_evals, hist_X
    
    # iteration of first critical solutions found + fitness values
    def get_first_critical(self):
        hist = self.history
        res = Population() 
        iter = 0
        if hist is not None:
            for algo in hist:
                iter += 1
                #n_evals.append(algo.evaluator.n_eval)  # store the number of function evaluations
                opt = algo.opt  # retrieve the optimum from the algorithm
                crit = np.where((opt.get("CB"))) [0] 
                feas = np.where((opt.get("feasible"))) [0] 
                feas = list(set(crit) & set(feas))
                res = opt[feas]
                if len(res) == 0:
                    continue
                else:
                    return iter, res
        return 0, res
    
    def obtain_history(self, critical=False):
        hist = self.history
        if hist is not None:
            n_evals = []  # corresponding number of function evaluations
            hist_F = []  # the objective space values in each generation
            for algo in hist:
                n_evals.append(algo.evaluator.n_eval)  # store the number of function evaluations
                opt = algo.opt  # retrieve the optimum from the algorithm
                if critical:
                    crit = np.where((opt.get("CB"))) [0] 
                    feas = np.where((opt.get("feasible"))) [0] 
                    feas = list(set(crit) & set(feas))
                else:
                    feas = np.where(opt.get("feasible"))[0]  # filter out only the feasible and append and objective space values
                hist_F.append(opt.get("F")[feas])
        else:
            n_evals = None
            hist_F = None
        return n_evals, hist_F

    def obtain_all_population(self):
        all_population = Population()
        hist = self.history
        for generation in hist:
            all_population = Population.merge(all_population, generation.pop)
        return all_population

    def obtain_history_hitherto(self, critical=False, nds=True):
        hist = self.history
        n_evals = []  # corresponding number of function evaluations
        hist_F = []  # the objective space values in each generation

        opt_all = Population()
        for algo in hist:
            n_evals.append(algo.evaluator.n_eval)
            opt_all = Population.merge(opt_all, algo.pop)  
            if nds:
                opt_all_nds = get_nondominated_population(opt_all)
            else:
                opt_all_nds = opt_all
            if critical:
                crit = np.where((opt_all_nds.get("CB"))) [0] 
                feas = np.where((opt_all_nds.get("feasible")))[0] 
                feas = list(set(crit) & set(feas))
            else:
                feas = np.where(opt_all_nds.get("feasible"))[0]  # filter out only the feasible and append and objective space values
            hist_F.append(opt_all_nds.get("F")[feas])
        return n_evals, hist_F
    
    def persist(self, save_folder):
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        with open(save_folder + os.sep + "result", "wb") as f:
            dill.dump(self, f)

    @staticmethod
    def load(save_folder, name="result"):
        with open(save_folder + os.sep + name, "rb") as f:
            return dill.load(f)
