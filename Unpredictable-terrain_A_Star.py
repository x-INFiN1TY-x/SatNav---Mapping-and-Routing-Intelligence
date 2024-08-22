import heapq
import math
import numpy as np

class Node:
    def __init__(self, position, parent=None, g=0, h=0):
        self.position = position
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f

def euclidean_distance(node_a, node_b):
    return math.sqrt((node_a.position[0] - node_b.position[0]) ** 2 + (node_a.position[1] - node_b.position[1]) ** 2)

def manhattan_distance(node_a, node_b):
    return abs(node_a.position[0] - node_b.position[0]) + abs(node_a.position[1] - node_b.position[1])

def is_feasible(node, binary_map):
    y, x = node.position
    return binary_map[y, x] == 1

def find_path(s_start, s_end, binary_map):
    rows, cols = len(binary_map), len(binary_map[0])
    open_list = []
    heapq.heappush(open_list, Node(s_start))
    closed_set = set()

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node.position == s_end:
            return reconstruct_path(current_node)

        closed_set.add(current_node.position)

        for movement in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            neighbor_pos = (current_node.position[0] + movement[0], current_node.position[1] + movement[1])

            if not (0 <= neighbor_pos[0] < rows and 0 <= neighbor_pos[1] < cols):
                continue

            neighbor_node = Node(neighbor_pos, current_node)

            if not is_feasible(neighbor_node, binary_map) or neighbor_pos in closed_set:
                continue

            neighbor_node.g = current_node.g + euclidean_distance(current_node, neighbor_node)
            neighbor_node.h = manhattan_distance(neighbor_node, Node(s_end))
            neighbor_node.f = neighbor_node.g + neighbor_node.h

            if neighbor_node.h < 100:
                for node in open_list:
                    if node.position == s_end:
                        return reconstruct_path(node)

            for open_node in open_list:
                if neighbor_node.position == open_node.position:
                    if neighbor_node.g < open_node.g:
                        open_list.remove(open_node)
                        heapq.heapify(open_list)
                    else:
                        break
            else:
                heapq.heappush(open_list, neighbor_node)

    return None

def reconstruct_path(node):
    path = []
    while node:
        path.append(node.position)
        node = node.parent
    return path[::-1]

def rough_terrain_a_star(s_start, s_end, binary_map):
    return find_path(s_start, s_end, binary_map)

