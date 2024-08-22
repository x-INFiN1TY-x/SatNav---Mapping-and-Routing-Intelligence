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

def standard_a_star(grid, src, dest):
    rows, cols = grid.shape
    open_list = []
    heapq.heappush(open_list, Node(src))
    closed_list = [[False for _ in range(cols)] for _ in range(rows)]

    cell_details = [[Node((i, j)) for j in range(cols)] for i in range(rows)]

    cell_details[src[0]][src[1]].g = 0
    cell_details[src[0]][src[1]].h = 0
    cell_details[src[0]][src[1]].f = 0

    while open_list:
        current_node = heapq.heappop(open_list)
        i, j = current_node.position

        closed_list[i][j] = True

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for dir in directions:
            new_i = i + dir[0]
            new_j = j + dir[1]

            if 0 <= new_i < rows and 0 <= new_j < cols and grid[new_i][new_j] == 1 and not closed_list[new_i][new_j]:
                if (new_i, new_j) == dest:
                    cell_details[new_i][new_j].parent = cell_details[i][j]
                    path = []
                    while cell_details[new_i][new_j].parent:
                        path.append((new_i, new_j))
                        new_i, new_j = cell_details[new_i][new_j].parent.position
                    path.append(src)
                    return path[::-1]

                g_new = cell_details[i][j].g + 1.0
                h_new = euclidean_distance(Node((new_i, new_j)), Node(dest))
                f_new = g_new + h_new

                if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                    heapq.heappush(open_list, Node((new_i, new_j), cell_details[i][j], g_new, h_new))
                    cell_details[new_i][new_j].f = f_new
                    cell_details[new_i][new_j].g = g_new
                    cell_details[new_i][new_j].h = h_new
                    cell_details[new_i][new_j].parent = cell_details[i][j]

    return None

