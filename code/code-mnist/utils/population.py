from evaluation.critical import Critical
from model_ga.population import PopulationExtended

def update_ind(population: PopulationExtended, crit_fnc: Critical):
    for ind in population:
        ind.set("CB", crit_fnc.eval(ind.get("F"),simout=None))
        ind.set("SO", None) # delete simout to save storage
    return population