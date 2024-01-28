from evaluation.fitness import *
from evaluation.critical import *

class CriticalMNISTMulti(Critical):
    def eval(self, vec_fit: np.ndarray, simout: SimulationOutput) -> bool:
        return vec_fit[0] < -0.5 and \
            vec_fit[1] < 0.2 #and \
            #vec_fit[2] < -3 and vec_fit[2] > -4   # distance between -2 and -4

class CriticalMNISTEntropy(Critical):
    def eval(self, vec_fit: np.ndarray, simout: SimulationOutput) -> bool:
        return vec_fit[0] < -0.7 and vec_fit[1] > 0.5 and vec_fit[1] < 0.55

class CriticalMNISTConf(Critical):
    ''' Conf_diff_max < 0'''
    def eval(self, vec_fit: np.ndarray, simout: SimulationOutput) -> bool:
        return vec_fit[0] < 0

class CriticalMNISTConf_05(Critical):
    def eval(self, vec_fit: np.ndarray, simout: SimulationOutput) -> bool:
        return vec_fit[0] < -0.5

class CriticalMNISTConf_07(Critical):
    def eval(self, vec_fit: np.ndarray, simout: SimulationOutput) -> bool:
        return vec_fit[0] < -0.7

class CriticalMNISTConf_09(Critical):
    def eval(self, vec_fit: np.ndarray, simout: SimulationOutput) -> bool:
        return vec_fit[0] < -0.9

class CriticalMNISTConf_095(Critical):
    def eval(self, vec_fit: np.ndarray, simout: SimulationOutput) -> bool:
        return vec_fit[0] < -0.95

class CriticalMNIST(Critical):
    ''' Conf_diff_max < 0, Conf_missclassified_label > 0.5'''
    def eval(self, vec_fit: np.ndarray, simout: SimulationOutput) -> bool:
        # return vec_fit[0] < 0 and vec_fit[1] < 0.1  # Conf diff, Conf miss
        # return vec_fit[0] < 0 and (vec_fit[1] == -7) # Conf diff, Predicted
        # return vec_fit[0] < -0.2 and vec_fit[1] < -1 # Conf diff, Dist archive # Random value, need some method/explanation to choose it.
        # return vec_fit[0] < 0 and (vec_fit[1] < -0.5) # Conf diff, maximize prediction as 8
        return vec_fit[0] < -0.5 and vec_fit[1] < 0.2 # Conf diff, Dist archive # Random value, need some method/explanation to choose it.
        # return vec_fit[0] < 0.5 and vec_fit[1] < 0.6 # Conf diff, Dist archive # Random value, need some method/explanation to choose it.
        #return vec_fit[0] < 0.2 and vec_fit[1] > 0.5 and vec_fit[1] < 0.55  # Conf diff, Dist archive # Random value, need some method/explanation to choose it.

''' Fitness function for MNIST Problem'''

class FitnessMNIST(Fitness):
    def __init__(self, diversify=False) -> None:
        super().__init__()
        self.diversify = diversify

    @property
    def min_or_max(self):
        if self.diversify:
            return "min", "min", "min"
        else:
            return "min", "min"

    @property
    def name(self):
        #name =  "Conf_diff_max", "Coverage" #"Distance_archive" #"Conf_miss" #"Predicted Label"
        #name = "Confidence_Expected", "Luminosity" #"Distance_archive" #"Conf_miss" #"Predicted Label"
        name = "Confidence_Expected", "Brightness", "Distance_Input" #"Distance_Archive" #"Distance_archive" #"Conf_miss" #"Predicted Label"
        #return "Entropy_Signed", "Coverage" #, "Distance_Archive" #"Distance_archive" #"Conf_miss" #"Predicted Label"
        if self.diversify:
            name = name[0:3]
        else:
            name = name[0:2]
        return name

    # HACK. needed to add data parameter, as information is not sotred in simout
    def eval(self, simout: SimulationOutput, data: Dict) -> Tuple[float]:   
        # Evalute fitness value of the classification ( = simulation)
        predicted_label = data["predicted_label"]
        confidence = data["confidence"]
        predictions = data["predictions"]
        expected_label = data["expected_label"]
        brightness = data["brightness"]
        distance_archive = data["distance_archive"]
        coverage = data["coverage"]
        move_distance = data["move_distance"]
        angle = data["angle"]
        entropy_signed = data["entropy_signed"]
        distance_test_input = data["distance_test_input"]
        coverage_rel = data["coverage_rel"]

        ''' Fitness 1 '''
        # Assign fitness value to individual 
        # ff1: as in DeepHyperion paper, ff2: mis predicted label confidence (min)
        ff1 = confidence #confidence if confidence > 0 else -0.1
        #ff1 = predictions[expected_label]
        #ff1 = entropy_signed

        ''' Fitness 2 '''
        # ff2 = predicted_label
        # ff2 = predictions[predicted_label] if expected_label != predicted_label else 0
        # ff2 = predictions[8] # confidence to get an eight
        # ff2 = distance_archive
        ff2 = brightness
        # ff2 = coverage
        # ff2 = coverage_rel
        #ff2 = angle

        # ff3 = -distance_archive
        ff3 = - distance_test_input 

        if self.diversify:
            f_vector = (ff1, ff2, ff3)
        else:
            f_vector = (ff1, ff2)
    
        return f_vector

''' Filtered fitness function for MNIST Problem'''
class FitnessMNISTFiltered(Fitness):
    @property
    def min_or_max(self):
        return "min", "min"

    @property
    def name(self):
        return "Conf_diff_max", "Coverage (neg)" #"Distance_archive" #"Conf_miss" #"Predicted Label"

    # HACK. needed to add data parameter, as information is not sotred in simout
    def eval(self, simout: SimulationOutput, data: Dict) -> Tuple[float]:
        # Evalute fitness value of the classification ( = simulation)
        predicted_label = data["predicted_label"]
        confidence = data["confidence"]
        predictions = data["predictions"]
        expected_label = data["expected_label"]
        #distance_archive = data["distance_archive"]
        coverage = data["coverage"]

        # Assign fitness value to individual 
        # ff1: as in DeepHyperion paper, ff2: mis predicted label confidence (min)
        ff1 = confidence #confidence if confidence > 0 else -0.1
        # ff2 = predicted_label
        # ff2 = predictions[predicted_label] if expected_label != predicted_label else 0
        #ff1 = predictions[expected_label]
        
        ff2 = coverage
        # ff2 = distance_archive
        # predictions[predicted_label] if predicted_label != expected_label \
        #        else 0vec_fit
        PENALTY_FF1 = 1
        PENALTY_FF2 = 0

        print("APPLYING MNIST FILTERED")

        if CriticalMNIST().eval([ff1,ff2], simout=None):
            return (ff1, ff2)
        else:
            return (PENALTY_FF1, PENALTY_FF2)