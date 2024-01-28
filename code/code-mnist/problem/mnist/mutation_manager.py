import random
import xml.etree.ElementTree as ET
import re
from random import randint, uniform
from problem.mnist.properties import MUTLOWERBOUND, MUTUPPERBOUND, MUTOFPROB
from utils.math_utils import round
from problem.mnist.properties import IMG_SIZE
from numpy import sign

NAMESPACE = '{http://www.w3.org/2000/svg}'


def apply_displacement_to_mutant(value, extent):
    
    result = float(value) + extent
    
    # if resulting location is out of bounds, 
    # move the point back by the amount it exceeds
    
    if result < 0:
        result = float(value) - result
    if result > IMG_SIZE:
        result = IMG_SIZE - (result - IMG_SIZE)
    
    assert ( result <= IMG_SIZE and result >= 0)

    # result = float(value) + 0.8 * extent
    # displ = uniform(MUTLOWERBOUND, MUTUPPERBOUND) * extent
    # if random.uniform(0, 1) >= MUTOFPROB:
    #     result = float(value) + displ
    # else:
    #     result = float(value) - displ
    return repr(result)

def apply_mutoperator_fix(svg_path, extent1, extent2):    
    # find all the vertexes
    pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
    segments = pattern.findall(svg_path)    
    # chose a random vertex
    num_matches = len(segments) * 2
    path = svg_path
    if num_matches > 1:  
        # apply to first and second vertex
        # random_coordinate_index = randint(0, num_matches - 1)
        svg_iter = re.finditer(pattern, svg_path)
        index1 = 1
        vertex = next(value for index, value in enumerate(svg_iter) if int(index == int(index1 / 2)))
        group_index1 = (index1 % 2) + 1
        value = apply_displacement_to_mutant(vertex.group(group_index1), extent1)
        path = svg_path[:vertex.start(group_index1)] + value + svg_path[vertex.end(group_index1):]
        
        index2 = 2
        vertex = next(value for index, value in enumerate(svg_iter) if int(index == int(index2 / 2)))
        group_index2 = (index2 % 2) + 1
        value = apply_displacement_to_mutant(vertex.group(group_index2), extent2)
        path = svg_path[:vertex.start(group_index2)] + value + svg_path[vertex.end(group_index2):]
    else:
        print("ERROR")
        print(svg_path)
    
    return path

def apply_mutoperator1(svg_path, extent, c_index=None):    
    # find all the vertexes
    pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
    segments = pattern.findall(svg_path)    
    # print(f"c_index: {c_index}")
    # print(f"Digit has {len(segments)} segments")

    # chose a random vertex
    num_matches = len(segments) * 2
    path = svg_path
    if num_matches > 0:  
        if c_index is not None:
            random_coordinate_index = c_index
        else:
            random_coordinate_index = randint(0, num_matches - 1)
        # print("---------------------")
        # print(f"num_matches: {num_matches}")
        # print(f"random_coordinate_index: {random_coordinate_index}")
        svg_iter = re.finditer(pattern, svg_path)
        
        # for index,value in enumerate(svg_iter):
        #     print(f"{index, value}")

        for index, value in enumerate(svg_iter):
            if int(index == int(random_coordinate_index / 2)):
                # print("vertex found")
                vertex = value
                break
        # vertex = next(value for index, value in enumerate(svg_iter) if int(index == int(random_coordinate_index / 2)))        
        # print(f"vertex selected: {vertex}")
        
        group_index = int((random_coordinate_index % 2) + 1)
        # print(f"group_index: {group_index}")
        # print(f"vertex.group(group_index): {vertex.group(group_index)}")

        value = apply_displacement_to_mutant(vertex.group(group_index), extent)
        path = svg_path[:vertex.start(group_index)] + value + svg_path[vertex.end(group_index):]
        #print(f"path_prefix: {svg_path[:vertex.start(group_index)]}")
    else:
        print("ERROR")
        print(svg_path)
    
    return path


def apply_mutoperator2_new(svg_path, extent, c_index=None):
    # find all the vertexes
    pattern = re.compile('C\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
    segments = pattern.findall(svg_path)
    # chose a random control point
    num_matches = len(segments) * 2 # user can select between two points in a segment
    # print(f"num_matches: {num_matches}")
    path = svg_path
    if num_matches > 0:
        if c_index is not None:
            random_coordinate_index = c_index % num_matches
        else:
            random_coordinate_index = randint(0, num_matches - 1)
        # print(f"random_coordinate_index: {random_coordinate_index}")
        svg_iter = re.finditer(pattern, svg_path)
        control_point = next(value for index, value in enumerate(svg_iter) if int(index == int(random_coordinate_index/2)))
        
        # print(f"control point: {control_point}")

        group_index = int((random_coordinate_index % 2) + 1)
        
        # print(f"group_index: {group_index}")
        # print(f"vertex.group(group_index): {control_point.group(group_index)}")

        value = apply_displacement_to_mutant(control_point.group(group_index), extent)

        path = svg_path[:control_point.start(group_index)] + value + svg_path[control_point.end(group_index):]
    else:
        print("ERROR")
        print(svg_path)
    return path

# def apply_mutoperator2(svg_path, extent, c_index=None):
#     # find all the vertexes
#     pattern = re.compile('C\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
#     segments = pattern.findall(svg_path)
#     # chose a random control point
#     num_matches = len(segments) * 4
#     path = svg_path
#     if num_matches > 0:
#         if c_index is not None:
#             random_coordinate_index = c_index
#         else:
#             random_coordinate_index = randint(0, num_matches - 1)
#         # print(f"random_coordinate_index: {random_coordinate_index}")
#         svg_iter = re.finditer(pattern, svg_path)
#         control_point = next(value for index, value in enumerate(svg_iter) if int(index == int(random_coordinate_index/4)))
#         group_index = (random_coordinate_index % 4) + 1
#         value = apply_displacement_to_mutant(control_point.group(group_index), extent)
#         path = svg_path[:control_point.start(group_index)] + value + svg_path[control_point.end(group_index):]
#     else:
#         print("ERROR")
#         print(svg_path)
#     return path


def apply_mutoperator3_new(svg_path, c_index=None):
    # find all the vertexes
    pattern = re.compile('C\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
    segments = pattern.findall(svg_path)
    # chose a random control point
    num_matches = len(segments)
    # print(len(segments))
    path = svg_path
    if num_matches > 3:
        if c_index is not None:
            random_coordinate_index = int(round(c_index /2)) % len(segments)
        else:
            random_coordinate_index = randint(0, num_matches - 1)
        # print(f"random_coordinate_index: {random_coordinate_index}")
        control_point = segments[random_coordinate_index]
        c_index = "C " + control_point[0] + ',' + control_point[1] + ' ' + control_point[2] + ',' + control_point[3] + ' ' + control_point[4] + ',' + control_point[5] + ' '
        # remove a control point from path
        path = re.sub(c_index,'', svg_path)
    else:
        print("ERROR")
        print(svg_path)
    return path


def apply_mutoperator4_new(svg_path, c_index=None):
    # find all the vertexes
    pattern = re.compile('C\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
    segments = pattern.findall(svg_path)
    # chose a random control point
    num_matches = len(segments)
    path = svg_path
    if num_matches > 2:
        while (True):
            random_coordinate_index = randint(0, num_matches - 1)
            if c_index is not None:
                random_coordinate_index = int(round(c_index /2))
            else:
                random_coordinate_index = randint(0, num_matches - 1)
            # print(f"random_coordinate_index: {random_coordinate_index}")
            # print(f"num_matches: {num_matches}")

            if random_coordinate_index + 1 <= num_matches -1:                
                segment = segments[random_coordinate_index]
                old_c_index = "C " + segment[0] + ',' + segment[1] + ' ' + segment[2] + ',' + segment[3] + ' ' + segment[4] + ',' + segment[5] + ' '
                c_index = "C " + segment[0] + ',' + segment[1] + ' ' + segment[2] + ',' + segment[3] + ' ' + str(random.uniform(float(segment[0]), float(segment[4]))) + ',' + str(random.uniform(float(segment[1]), float(segment[5]))) + ' '
                next_segment = segments[random_coordinate_index + 1]
                new_c_index = "C " + str(random.uniform(float(segment[2]), float(segment[4]))) + ',' + str(random.uniform(float(segment[3]), float(segment[5]))) + ' ' + str(random.uniform(float(segment[4]), float(next_segment[0]))) + ',' + str(random.uniform(float(segment[5]), float(next_segment[1]))) + ' ' + str(random.uniform(float(segment[4]), float(next_segment[4]))) + ',' + str(random.uniform(float(segment[5]), float(next_segment[5]))) + ' '
                path = re.sub(old_c_index, c_index + new_c_index, svg_path)
                break
            else:
                continue
    else:
        print("ERROR")
        print(svg_path)
    return path


# def apply_mutoperator3(svg_path):
#     # find all the vertexes
#     pattern = re.compile('C\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
#     segments = pattern.findall(svg_path)
#     # chose a random control point
#     num_matches = len(segments)
#     path = svg_path
#     if num_matches > 3:
#         random_coordinate_index = randint(0, num_matches - 1)
#         control_point = segments[random_coordinate_index]
#         cp = "C " + control_point[0] + ',' + control_point[1] + ' ' + control_point[2] + ',' + control_point[3] + ' ' + control_point[4] + ',' + control_point[5] + ' '
#         # remove a control point from path
#         path = re.sub(cp,'', svg_path)
#     else:
#         print("ERROR")
#         print(svg_path)
#     return path


# def apply_mutoperator4(svg_path):
#     # find all the vertexes
#     pattern = re.compile('C\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
#     segments = pattern.findall(svg_path)
#     # chose a random control point
#     num_matches = len(segments)
#     path = svg_path
#     if num_matches > 2:
#         while (True):
#             random_coordinate_index = randint(0, num_matches - 1)
#             if random_coordinate_index + 1 <= num_matches -1:                
#                 segment = segments[random_coordinate_index]
#                 old_cp = "C " + segment[0] + ',' + segment[1] + ' ' + segment[2] + ',' + segment[3] + ' ' + segment[4] + ',' + segment[5] + ' '
#                 cp = "C " + segment[0] + ',' + segment[1] + ' ' + segment[2] + ',' + segment[3] + ' ' + str(random.uniform(float(segment[0]), float(segment[4]))) + ',' + str(random.uniform(float(segment[1]), float(segment[5]))) + ' '
#                 next_segment = segments[random_coordinate_index + 1]
#                 new_cp = "C " + str(random.uniform(float(segment[2]), float(segment[4]))) + ',' + str(random.uniform(float(segment[3]), float(segment[5]))) + ' ' + str(random.uniform(float(segment[4]), float(next_segment[0]))) + ',' + str(random.uniform(float(segment[5]), float(next_segment[1]))) + ' ' + str(random.uniform(float(segment[4]), float(next_segment[4]))) + ',' + str(random.uniform(float(segment[5]), float(next_segment[5]))) + ' '
#                 path = re.sub(old_cp, cp + new_cp, svg_path)
#                 break
#             else:
#                 continue
#     else:
#         print("ERROR")
#         print(svg_path)
#     return path

def mutate_series(svg_desc, extent_1, extent_2):
    root = ET.fromstring(svg_desc)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    path_1 = apply_mutoperator1(svg_path, extent_1)
    path_2 = apply_mutoperator1(path_1, extent_2)  
    return path_2

def mutate_point(svg_desc, operator_name, extent_1, extent_2, segment=None):
    root = ET.fromstring(svg_desc)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    mutant_vector = svg_path    
    if operator_name == 1:
        mutant_vector = apply_mutoperator1_point(svg_path,  extent_1, extent_2, segment=segment)
    elif operator_name == 2:
        mutant_vector = apply_mutoperator2_point(svg_path,  extent_1, extent_2, segment=segment)  
    return mutant_vector

def apply_mutoperator2_point(svg_path,  extent_1, extent_2, segment=None):
    # find all the vertexes
    pattern = re.compile('C\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
    segments = pattern.findall(svg_path)    
    # chose a random vertex
    num_matches = len(segments)
    path = svg_path
    if num_matches > 0:  
        if segment is not None:
            random_segment = segment % len(segments) # bug, so that segment number varies
        else:
            random_segment = randint(0, num_matches - 1)
        svg_iter = re.finditer(pattern, svg_path)
        
        for index, value in enumerate(svg_iter):
            if int(index == int(random_segment )):
                # print("vertex found")
                vertex = value
                break
        group_index_1 = 1
        # print(f"group_index: {group_index_1}")
        # print(f"vertex.group(group_index): {vertex.group(group_index_1)}")
        value = apply_displacement_to_mutant(vertex.group(group_index_1), extent_1)
        path = svg_path[:vertex.start(group_index_1)] + value + svg_path[vertex.end(group_index_1):]
        
        group_index_2 = 2
        # print(f"group_index: {group_index_2}")
        # print(f"vertex.group(group_index): {vertex.group(group_index_2)}")
        value = apply_displacement_to_mutant(vertex.group(group_index_2), extent_2)
        path = path[:vertex.start(group_index_2)] + value + path[vertex.end(group_index_2):]
    else:
        print("ERROR")
        print(svg_path)
    
    return path
def apply_mutoperator1_point(svg_path,  extent_1, extent_2, segment=None):
    # find all the vertexes
    pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
    segments = pattern.findall(svg_path)    
    # print(f"Provided segment: {segment}")

    # print(f"Digit has {len(segments)} segments")
    # chose a random vertex
    num_matches = len(segments)
    path = svg_path
    if num_matches > 0:  
        if segment is not None:
            random_segment = segment % len(segments) 
        else:
            random_segment = randint(0, num_matches - 1)
        svg_iter = re.finditer(pattern, svg_path)
        
        for index, value in enumerate(svg_iter):
            if int(index == int(random_segment )):
                # print("vertex found")
                vertex = value
                break
        group_index_1 = 1
        print(f"group_index: {group_index_1}")
        print(f"vertex.group(group_index): {vertex.group(group_index_1)}")

        value = apply_displacement_to_mutant(vertex.group(group_index_1), extent_1)
        path = svg_path[:vertex.start(group_index_1)] + value + svg_path[vertex.end(group_index_1):]
        
        group_index_2 = 2
        print(f"group_index: {group_index_2}")
        print(f"vertex.group(group_index): {vertex.group(group_index_2)}")
        value = apply_displacement_to_mutant(vertex.group(group_index_2), extent_2)
        path = path[:vertex.start(group_index_2)] + value + path[vertex.end(group_index_2):]
    else:
        print("ERROR")
        print(svg_path)
    
    return path

def mutate(svg_desc, operator_name, mutation_extent, c_index=None):
    root = ET.fromstring(svg_desc)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    mutant_vector = svg_path    
    if operator_name == 1:
        mutant_vector = apply_mutoperator1(svg_path, mutation_extent, c_index)
    elif operator_name == 2:        
        mutant_vector = apply_mutoperator2_new(svg_path, mutation_extent, c_index)  
    elif operator_name == 3:        
        mutant_vector = apply_mutoperator3_new(svg_path, c_index = c_index)
    elif operator_name == 4:        
        mutant_vector = apply_mutoperator4_new(svg_path, c_index = c_index)
    # elif operator_name == 5:        
    #     mutant_vector = apply_mutoperator5(svg_path, c_index = c_index)
    return mutant_vector

def mutate_fix(svg_desc, extent_1, extent_2):
    root = ET.fromstring(svg_desc)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    mutant_vector = apply_mutoperator_fix(svg_path, extent_1, extent_2)
    return mutant_vector

def mutate_op1(svg_desc, mutation_extent):
    root = ET.fromstring(svg_desc)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    mutant_vector = apply_mutoperator1(svg_path, mutation_extent)
    return mutant_vector