import heapq
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def preprocess_image(image, threshold=127):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_map = cv2.threshold(grayscale, threshold, 1, cv2.THRESH_BINARY)
    return binary_map

def visualize_path(image, path, start, end):
    vis_image = image.copy()
    
    for i, (x, y) in enumerate(path):
        color = (0, 255 - int(255 * i / len(path)), int(255 * i / len(path)))
        cv2.circle(vis_image, (y, x), 5, color, -1)
    
    cv2.circle(vis_image, (start[1], start[0]), 10, (0, 255, 0), -1)
    cv2.circle(vis_image, (end[1], end[0]), 10, (0, 0, 255), -1)
    
    cv2.putText(vis_image, 'Start', (start[1] + 15, start[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(vis_image, 'End', (end[1] + 15, end[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.title("Path Visualization")
    plt.axis('off')
    plt.show()

def visualize_grid(grid):
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='binary')
    plt.title("Binary Grid Representation")
    plt.colorbar(label='Walkable (1) / Obstacle (0)')
    plt.show()

def select_points(event, x, y, flags, param):
    points = param
    if event == cv2.EVENT_LBUTTONDOWN:
        if points['start'] is None:
            points['start'] = (y, x)
            print(f"Start point selected at: {points['start']}")
        elif points['end'] is None:
            points['end'] = (y, x)
            print(f"End point selected at: {points['end']}")

def run_pathfinding(image):
    points = {'start': None, 'end': None}
    
    cv2.imshow('Map', image)
    cv2.setMouseCallback('Map', select_points, points)
    print("Click on the image to set the start point and end point.")

    while points['end'] is None:
        if cv2.waitKey(1) & 0xFF == 27:
            print("Selection cancelled.")
            cv2.destroyAllWindows()
            return
    cv2.destroyAllWindows()
    
    src = points['start']
    dest = points['end']

    grid = preprocess_image(image)
    
    visualize_grid(grid)
    
    blocked_percentage = calculate_blocked_percentage(grid)
    print(f"Blocked cell percentage: {blocked_percentage:.2f}%")

    if blocked_percentage > 20:
        print("Using standard A* for construction-heavy area.")
        path = standard_a_star(grid, src, dest)
    else:
        print("Using rough-terrain A* for less construction-heavy terrain.")
        path = rough_terrain_a_star(src, dest, grid)

    if path:
        print(f"Path found with {len(path)} steps.")
        visualize_path(image, path, src, dest)
    else:
        print("No path found.")

def calculate_blocked_percentage(grid):
    total_cells = grid.size
    blocked_cells = np.sum(grid == 0)
    return (blocked_cells / total_cells) * 100

if __name__ == "__main__":
    image = cv2.imread('map.png')
    run_pathfinding(image)
