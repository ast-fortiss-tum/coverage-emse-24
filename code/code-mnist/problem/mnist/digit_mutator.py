import random
from problem.mnist import mutation_manager
from problem.mnist import rasterization_tools
from problem.mnist import vectorization_tools
from problem.mnist.properties import MUTOPPROB, MUTOFPROB, DISTANCE, TSHD_TYPE, DISTANCE_SEED
from problem.mnist.utils_mnist import get_distance, reshape
from problem.mnist_loader import mnist_loader

TSHD_TYPE = '1'
MUTOPPROB = 0.5
MUTOFPROB = MUTOPPROB

class DigitMutator:
    def __init__(self, digit):
        self.digit = digit
        
    # To make tests reproducible
    # random.seed(digit.seed)
    def mutate_point(self, 
                        extent_x=None, 
                        extent_y = None, 
                        segment=None, 
                        mutation=None):

            counter_mutations = 0

            if mutation is None:
                mutation = 1
                
            counter_mutations += 1

            if extent_x or extent_y is None:
                extent_x = counter_mutations/20
                extent_y = extent_x

            print(f"read segment: {segment}")
            mutant_vector = mutation_manager.mutate_point(
                                    self.digit.xml_desc, 
                                    mutation, 
                                    extent_1=extent_x,
                                    extent_2=extent_y,
                                    segment=segment)
                
            mutant_xml_desc = vectorization_tools.create_svg_xml(mutant_vector)
            rasterized_digit = rasterization_tools.rasterize_in_memory(mutant_xml_desc)
            
            seed_image = mnist_loader.get_x_test()[int(self.digit.seed)]
            xml_desc = vectorization_tools.vectorize(seed_image)
            seed = rasterization_tools.rasterize_in_memory(xml_desc)
            distance_seed = get_distance(seed, rasterized_digit)

            print(f"distance seed: {distance_seed}")
            self.digit.xml_desc = mutant_xml_desc
            self.digit.purified = rasterized_digit
            self.digit.predicted_label = None
            self.digit.confidence = None

    def mutate(self, extent=None, c_index=None, mutation=None):
        condition = True
        counter_mutations = 0
        while condition:
            # Select mutation operator.
            # rand_mutation_probability = random.uniform(0, 1)
            # rand_mutation_prob = random.uniform(0, 1)
            # if rand_mutation_probability >= MUTOPPROB:            
            #     if rand_mutation_prob >= MUTOFPROB:
            #         mutation = 1
            #     else:
            #         mutation = 2
            # else:
            #     if rand_mutation_prob >= MUTOFPROB:
            #         mutation = 3
            #     else:
            #         mutation = 4
            if mutation is None:
                mutation = 2
                
            counter_mutations += 1

            if extent is None:
                extent = counter_mutations/20
            
            mutant_vector = mutation_manager.mutate(self.digit.xml_desc, 
                                    mutation, 
                                    extent,
                                    c_index=c_index)
                
            mutant_xml_desc = vectorization_tools.create_svg_xml(mutant_vector)
            rasterized_digit = rasterization_tools.rasterize_in_memory(mutant_xml_desc)
            condition = False
            
            # distance_inputs = get_distance(self.digit.purified, rasterized_digit)
            # if (TSHD_TYPE == '0'):
            #     if distance_inputs != 0:
            #         condition = False
            # elif (TSHD_TYPE == '1'):
            #     seed_image = DigitMutator.x_test[int(self.digit.seed)]
            #     xml_desc = vectorization_tools.vectorize(seed_image)
            #     seed = rasterization_tools.rasterize_in_memory(xml_desc)
            #     distance_seed = get_distance(seed, rasterized_digit)
            #     # if distance_inputs != 0 and distance_seed <= DISTANCE and distance_seed != 0:
            #     condition = False
            #     # print("Repeating mutation.")
            # elif (TSHD_TYPE == '2'):
            #     seed = reshape(DigitMutator.x_test[int(self.digit.seed)])
            #     distance_seed = get_distance(seed, rasterized_digit)
            #     if distance_inputs != 0 and distance_seed <= DISTANCE_SEED and distance_seed != 0:
            #         condition = False
     
        self.digit.xml_desc = mutant_xml_desc
        self.digit.purified = rasterized_digit
        self.digit.predicted_label = None
        self.digit.confidence = None
    
    def mutate_fix(self,extent_1, extent_2):
        condition = True
        while condition:
            mutant_vector = mutation_manager.mutate_fix(self.digit.xml_desc, extent_1, extent_2)
            mutant_xml_desc = vectorization_tools.create_svg_xml(mutant_vector)
            rasterized_digit = rasterization_tools.rasterize_in_memory(mutant_xml_desc)

            distance_inputs = get_distance(self.digit.purified, rasterized_digit)
            
            # print(f"distance_inputs: {distance_inputs}")
            if distance_inputs != 0:
                condition = False
            else:
                print("Distance between mutated and original images is 0.")
                print("Repeating mutation...")
    
        self.digit.xml_desc = mutant_xml_desc
        self.digit.purified = rasterized_digit
        self.digit.predicted_label = None
        self.digit.confidence = None

    def mutate_op1(self,extent_1, extent_2):
        condition = True
        while condition:
            mutant_vector = mutation_manager.mutate_op1(self.digit.xml_desc, extent_1)
            mutant_xml_desc = vectorization_tools.create_svg_xml(mutant_vector)
            rasterized_digit = rasterization_tools.rasterize_in_memory(mutant_xml_desc)

            distance_inputs = get_distance(self.digit.purified, rasterized_digit)
            
            # print(f"distance_inputs: {distance_inputs}")
            if distance_inputs != 0:
                condition = False
            else:
                print("Distance between mutated and original images is 0.")
                print("Repeating mutation...")
    
        self.digit.xml_desc = mutant_xml_desc
        self.digit.purified = rasterized_digit
        self.digit.predicted_label = None
        self.digit.confidence = None

    def mutate_series(self,extent_1, extent_2):
        condition = True
        counter_mutations = 0
        while condition:
            mutant_vector = mutation_manager.mutate_series(self.digit.xml_desc, extent_1, extent_2)
            mutant_xml_desc = vectorization_tools.create_svg_xml(mutant_vector)
            rasterized_digit = rasterization_tools.rasterize_in_memory(mutant_xml_desc)

            distance_inputs = get_distance(self.digit.purified, rasterized_digit)
            
            print(f"distance_inputs: {distance_inputs}")
            if distance_inputs != 0:
                condition = False
    
        self.digit.xml_desc = mutant_xml_desc
        self.digit.purified = rasterized_digit
        self.digit.predicted_label = None
        self.digit.confidence = None

