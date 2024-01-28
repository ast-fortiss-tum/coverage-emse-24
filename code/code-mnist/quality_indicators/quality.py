from ast import List
from numpy import NaN
from pymoo.indicators.hv import *
from pymoo.indicators.igd import *
from pymoo.indicators.gd import *
from pymoo.indicators.gd_plus import *
from pymoo.indicators.igd_plus import *
from quality_indicators.metrics.cid import CID
from quality_indicators.metrics.spread import *
from quality_indicators.metrics import digd
from dataclasses import dataclass
import dill
import os
from pathlib import Path
import logging as log
from utils.sampling import CartesianSampling


class Quality(object):
    @staticmethod
    def calculate_cid(result, reference_set,  n_evals_by_axis): 
        hist = result.history
        problem = result.problem
        n_evals = []  # corresponding number of function evaluations
        hist_X_labeled_hitherto = []
        last_x_sofar  = np.asarray([])
        
        if hist is not None:
            #n_evals_all = 0
            for algo in hist:
                # store the number of function evaluations
                #n_evals_all = n_evals_all + algo.evaluator.n_eval
                n_evals.append( algo.evaluator.n_eval)
                # retrieve the optimum from the algorithm
                pop = algo.pop
                # filter out only the feasible and append and objective space values
                feas = np.where(pop.get("feasible"))[0]
                pop_feas = pop[feas]
                labels = np.where(pop_feas.get("CB"))[0]
                # critical individuals available
                if len(labels) > 0:
                    pop_labeled = pop_feas[labels]
                    pop_labeled_x = pop_labeled.get("X")
                    if len(last_x_sofar) > 0:
                        last_x_sofar = np.concatenate((last_x_sofar, pop_labeled_x), axis=0)
                    else:
                        last_x_sofar = pop_labeled_x

                hist_X_labeled_hitherto.append(last_x_sofar)

            if reference_set is None:
                reference_set = CartesianSampling(problem, n_evals_by_axis)

            metric_cid = CID(reference_set.get("X"), zero_to_one=True)

            # if X is empty (no critical solutions found) we need to return a different value as we cannot calculate the distance
            def get_cid_value(X):
                if len(X) == 0:
                    return 1
                else:
                    return metric_cid.do(X)
            
            cid = [get_cid_value(_X) for _X in hist_X_labeled_hitherto]

            return EvaluationResult("cid", n_evals, cid)
        else:
            return None
        
    @staticmethod
    def calculate_digd(result, input_crit):
        res = result
        problem = res.problem
        hist = res.history
        if hist is not None:
            n_evals, hist_X = res.obtain_history_design()
            digd_v = [ digd.digd(input_crit, 
                            X, 
                            lb = problem.xl, 
                            ub = problem.xu) 
                     for X in hist_X]  
            return EvaluationResult("digd", n_evals, digd_v)
        else:
            return None

    @staticmethod
    def calculate_hv(result, ref_point = None):
        res = result
        problem = res.problem
        hist = res.history
        if hist is not None:
            n_evals, hist_F = res.obtain_history()
            F = res.opt.get("F")
            approx_ideal = F.min(axis=0)
            approx_nadir = F.max(axis=0)
            n_obj = problem.n_obj
            if ref_point is None:
                ref_point = np.array(n_obj * [1.01])
            metric_hv = Hypervolume(ref_point=ref_point,
                                    norm_ref_point=False,
                                    zero_to_one=True,
                                    ideal=approx_ideal,
                                    nadir=approx_nadir)

            hv = [metric_hv.do(_F) for _F in hist_F]
            return EvaluationResult("hv", n_evals, hv)
        else:
            return None

    @staticmethod
    def calculate_hv_hitherto(result, critical_only =False, ref_point = None, ideal = None, nadir = None):
        res = result
        problem = res.problem
        hist = res.history
        if hist is not None:
            n_evals, hist_F = res.obtain_history_hitherto(critical_only)
            F = res.opt.get("F")            
            if ideal is None:
                ideal = F.min(axis=0)
            if nadir is None:
                nadir = F.max(axis=0)
            n_obj = problem.n_obj
            if ref_point is None:
                ref_point = np.array(n_obj * [1.01])
                norm_ref_point = False
            else:                  
                log.info("Reference point is given.")
                norm_ref_point = True
            metric_hv = Hypervolume(ref_point=ref_point,
                                    norm_ref_point=norm_ref_point,
                                    zero_to_one=True,
                                    ideal=ideal,
                                    nadir=nadir)

            hv = [metric_hv.do(_F) for _F in hist_F]
            return EvaluationResult("hv_global", n_evals, hv)
        else:
            return None

    @staticmethod
    def calculate_gd(result, input_pf=None, critical_only = False, mode='default'):
        res = result
        hist = res.history
        problem = res.problem        
        # provide a pareto front or use a pareto front from other sources
        if input_pf is not None:
            pf = input_pf
        else:
            pf = problem.pareto_front_n_points()
        hist = res.history
        if hist is not None:
            n_evals, hist_F = res.obtain_history(critical=critical_only)
            # log.info(hist_F)
            # log.info(pf)
            if pf is not None and len(pf) > 0:
                if mode == 'default':
                    metric_gd = GD(pf, zero_to_one=True)
                elif mode == 'plus':
                    metric_gd = GDPlus(pf, zero_to_one=True)
                else:
                    log.info(mode + " GD mode is not known. The default IGD is used.")
                    metric_gd = GD(pf, zero_to_one=True)
                gd = [metric_gd.do(_F) for _F in hist_F]
                return EvaluationResult("gd", n_evals, gd)
            else:
                log.info("No convergence analysis possible. The Pareto front is not known.")
                return None
        else:
            log.info("No convergence analysis possible. The history of the run is not given.")
            return None

    @staticmethod
    def calculate_gd_hitherto(result, input_pf=None, mode='default'):
        res = result
        hist = res.history
        problem = res.problem
        # provide a pareto front or use a pareto front from other sources
        if input_pf is not None:
            pf = input_pf
        else:
            pf = problem.pareto_front_n_points()
        hist = res.history
        if hist is not None:
            n_evals, hist_F = res.obtain_history_hitherto()
            if pf is not None and len(pf) > 0:
                if mode == 'default':
                    metric_gd = GD(pf, zero_to_one=True)
                elif mode == 'plus':
                    metric_gd = GDPlus(pf, zero_to_one=True)
                else:
                    log.info(mode + " GD mode is not known. The default IGD is used.")
                    metric_gd = GD(pf, zero_to_one=True)
                gd = [metric_gd.do(_F) for _F in hist_F]
                return EvaluationResult("gd_global", n_evals, gd)
            else:
                log.info("No convergence analysis possible. The Pareto front is not known.")
                return None
        else:
            log.info("No convergence analysis possible. The history of the run is not given.")
            return None
        
    @staticmethod
    def calculate_igd(result, critical_only = False, input_pf=None):
        res = result
        hist = res.history
        problem = res.problem
        # provide a pareto front or use a pareto front from other sources
        if input_pf is not None:
            pf = input_pf
        else:
            pf = problem.pareto_front_n_points()
        hist = res.history
        if hist is not None:
            n_evals, hist_F = res.obtain_history(critical=critical_only)
            if pf is not None:
                metric_igd = IGD(pf, zero_to_one=True)
                igd = [metric_igd.do(_F) for _F in hist_F]
                return EvaluationResult("igd", n_evals, igd)
            else:
                log.info("No convergence analysis possible. The Pareto front is not known.")
                return None
        else:
            log.info("No convergence analysis possible. The history of the run is not given.")
            return None

    @staticmethod
    def calculate_igd_hitherto(result, input_pf=None):
        res = result
        hist = res.history
        problem = res.problem
        # provide a pareto front or use a pareto front from other sources
        if input_pf is not None:
            pf = input_pf
        else:
            pf = problem.pareto_front_n_points()
        hist = res.history
        if hist is not None:
            n_evals, hist_F = res.obtain_history_hitherto()
            if pf is not None:
                metric_igd = IGD(pf, zero_to_one=True)
                igd = [metric_igd.do(_F) for _F in hist_F]
                return EvaluationResult("igd_global", n_evals, igd)
            else:
                log.info("No convergence analysis possible. The Pareto front is not known.")
                return None
        else:
            log.info("No convergence analysis possible. The history of the run is not given.")
            return None

    @staticmethod
    def calculate_sp(result, critical_only=False):
        res = result
        hist = res.history
        problem = res.problem

        # if problem.n_obj > 2:
        #     log.info("Uniformity Delta metric is only available for a 2D objective space.")
        #     return 0
        if hist is not None:
            n_evals, hist_F = res.obtain_history(critical=critical_only)
            uni = [spread(_F) for _F in hist_F]
            return EvaluationResult("sp", n_evals, uni)
        else:
            log.info("No uniformity analysis possible. The history of the run is not given.")
            return None

    @staticmethod
    def calculate_sp_hitherto(result):
        res = result
        hist = res.history
        problem = res.problem

        if problem.n_obj > 2:
            log.info("Uniformity Delta metric is only available for a 2D objective space.")
            return 0
        if hist is not None:
            n_evals, hist_F = res.obtain_history_hitherto()
            uni = [spread(_F) for _F in hist_F]
            return EvaluationResult("sp_global", n_evals, uni)
        else:
            log.info("No uniformity analysis possible. The history of the run is not given.")
            return None


@dataclass
class EvaluationResult(object):
    name: str
    steps: List(float)
    values: List(float)

    @staticmethod
    def load(save_folder, name):
        with open(save_folder + os.sep + name, "rb") as f:
            return dill.load(f)

    def persist(self, save_folder: str):
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        with open(save_folder + os.sep + self.name, "wb") as f:
            dill.dump(self, f)

    def to_string(self):
        return "name: " + str(self.name) + "\nsteps: " + str(self.steps) + "\nvalues: " + str(self.values)
