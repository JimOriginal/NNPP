import numpy as np
import random
import math
import torch
# Define the size of the map
map_size = (256, 256)

# Define the step size for the RRT* algorithm
step_size = 3

# Define the maximum number of iterations for the RRT* algorithm
max_iter = 1000000

# Define the radius of the ball for the RRT* algorithm

# Define the Euclidean distance function
def euclidean_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Define a function to check if a point is within the bounds of the map
def is_valid_point(p,map):
    return (p[0] >= 0 and p[0] < map_size[0] and p[1] >= 0 and p[1] < map_size[1] and map[p[0], p[1]] == 0.)

# Define a function to check if a point is obstacle-free
def is_free_point(p, map):
    return (map[p[0], p[1]] == 0.)

def is_free_point_underNN(p, map_nn):
    return (map_nn[p[0], p[1]] >=0.001)

# Define a function to generate a random point within the bounds of the map
def get_random_point(goal):
    if random.random() < 0.3:
        xrand = goal[0]
        yrand = goal[1]
    else:
        xrand = random.randint(0, map_size[0]-1)
        yrand = random.randint(0, map_size[1]-1)
    return (xrand, yrand)
 #   return (random.randint(0, map_size[0]-1), random.randint(0, map_size[1]-1))


def get_random_index(tensor):
    # Flatten tensor to a 1D array
    flat_tensor = tensor.flatten()
    
    # Compute normalized probability distribution
    prob_dist = flat_tensor / torch.sum(flat_tensor)
    prob_dist=prob_dist.detach().numpy()
    # Randomly sample index based on the probability distribution
    idx = np.random.choice(len(prob_dist), p=prob_dist)
    
    # Convert index to row and column coordinates in original tensor
    row_idx = idx // tensor.shape[1]
    col_idx = idx % tensor.shape[1]
    
    return (row_idx, col_idx)


def get_random_point_underNN(goal,map_nn):
    if random.random() > 0.5:
        return get_random_index(map_nn)
    elif random.random() < 0.1:
        xrand = goal[0]
        yrand = goal[1]
        return (xrand, yrand)
    else:
        return (random.randint(0, map_size[0]-1), random.randint(0, map_size[1]-1))

def get_random_point_useOnlyNN(map_nn):
    return get_random_index(map_nn)


# Define a function to find the nearest node to a given point
def find_nearest_node(p, nodes):
    dists = [euclidean_dist(p, n) for n in nodes]
    return nodes[np.argmin(dists)]

def product_near(tree_list, xrand):
    m = np.inf
    for i in range(0, len(tree_list)):
        if abs(tree_list[i][0] - xrand[0]) + abs(tree_list[i][1] - xrand[1]) < m:
            m = abs(tree_list[i][0] - xrand[0]) + abs(tree_list[i][1] - xrand[1])
            xnear = [tree_list[i][0], tree_list[i][1]]
    return xnear

# Define a function to generate a new node along the path between two existing nodes
def generate_new_node(near, rand):
    dist = euclidean_dist(near, rand)
    if dist < step_size:
        return rand
    else:
        theta = np.arctan2(rand[1]-near[1], rand[0]-near[0])
        x= near[0] + step_size*np.cos(theta)
        y= near[1] + step_size*np.sin(theta)
        x=int(x)
        y=int(y)
        return [x, y]

# Define a function to check if a path between two points is obstacle-free
def is_free_path(p1, p2, map):
    dist = euclidean_dist(p1, p2)
    # if dist < step_size:
    #     return is_free_point(p2, map)
    # else:
    theta = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
    num_steps = int(dist / step_size)
    for i in range(num_steps):

        p = (p1[0] + math.floor(i*step_size*np.cos(theta)), p1[1] + math.floor(i*step_size*np.sin(theta)))

        if not is_free_point(p, map):
            return False
    return True
def is_free_path_nn(p1, p2, map):
    dist = euclidean_dist(p1, p2)
    # if dist < step_size:
    #     return is_free_point(p2, map)
    # else:
    theta = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
    num_steps = int(dist / step_size)
    for i in range(num_steps):

        p = (p1[0] + math.floor(i*step_size*np.cos(theta)), p1[1] + math.floor(i*step_size*np.sin(theta)))

        if not is_free_point_underNN(p, map):
            return False
    return True

def rrt_search(map_dia,start,goal,max_iter=100000, to_torch=True):
    h, w = map_dia.shape
    start, goal = start.tolist(), goal.tolist()
    tree = [[start[0], start[1],0,0]]
    for i in range(max_iter):
        rand = get_random_point(goal)
        near = product_near(tree, rand)
        new_node = generate_new_node(near, rand)
        #print("new_node: ", new_node)
        if not is_valid_point(new_node,map_dia):
            continue
        if not is_free_path(near, new_node, map_dia):
            continue
        tree.append([new_node[0],new_node[1],near[0],near[1]])
        if np.linalg.norm(np.array(new_node) - np.array(goal))<2:
            print("goal found, final node", new_node)
            break
    tree_list=np.array(tree)
    rrt_path=[goal]
    n=len(tree_list)-1
    x = tree_list[n,0]
    y = tree_list[n,1]
    f_x = tree_list[n,2]
    f_y = tree_list[n,3]
    rrt_path.append([x,y])
    search_list=[]
    while start not in rrt_path:
        search_list = tree_list[np.where((tree_list[:,0]==f_x) & (tree_list[:,1]==f_y))][0]
        search_list = search_list.tolist()
        rrt_path.append([search_list[0],search_list[1]])
        f_x = search_list[2]
        f_y = search_list[3]
    rrt_path=linear_interpolation(rrt_path)
    if to_torch:
        rrt_path = torch.tensor(rrt_path)
    return rrt_path   

def bresenham(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    err = dx - dy

    points = []
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points
def linear_interpolation(points):
    interpolated_points = []
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        interpolated_points += bresenham(x0, y0, x1, y1)
    return interpolated_points


def rrt_search_underNN(map_dia,nn_map,start,goal,max_iter=100000, to_torch=True):
    h, w = map_dia.shape
    start, goal = start.tolist(), goal.tolist()
    tree = [[start[0], start[1],0,0]]
    for i in range(max_iter):
        rand = get_random_point_underNN(goal,nn_map)
        #rand = get_random_point_useOnlyNN(nn_map)
        near = product_near(tree, rand)
        new_node = generate_new_node(near, rand)
        #print("new_node: ", new_node)
        if not is_valid_point(new_node,map_dia):
            continue
        if not is_free_path(near, new_node, map_dia):
            continue
        tree.append([new_node[0],new_node[1],near[0],near[1]])
        if np.linalg.norm(np.array(new_node) - np.array(goal))<2:
            #print("goal found, final node", new_node)
            break
    tree_list=np.array(tree)
    rrt_path=[goal]
    n=len(tree_list)-1
    x = tree_list[n,0]
    y = tree_list[n,1]
    f_x = tree_list[n,2]
    f_y = tree_list[n,3]
    rrt_path.append([x,y])
    search_list=[]
    while start not in rrt_path:
        search_list = tree_list[np.where((tree_list[:,0]==f_x) & (tree_list[:,1]==f_y))][0]
        search_list = search_list.tolist()
        rrt_path.append([search_list[0],search_list[1]])
        f_x = search_list[2]
        f_y = search_list[3]
    rrt_path=linear_interpolation(rrt_path)
    if to_torch:
        rrt_path = torch.tensor(rrt_path)
    return rrt_path 

def rrt_search_OnlyunderNN(nn_map,start,goal,max_iter=100000, to_torch=True):
    start, goal = start.tolist(), goal.tolist()
    tree = [[start[0], start[1],0,0]]
    for i in range(max_iter):
        #rand = get_random_point_underNN(goal,nn_map)
        rand = get_random_point_useOnlyNN(nn_map)
        near = product_near(tree, rand)
        new_node = generate_new_node(near, rand)
        #print("new_node: ", new_node)
        if not is_valid_point(new_node,nn_map):
            continue
        if not is_free_path_nn(near, new_node, nn_map):
            continue
        tree.append([new_node[0],new_node[1],near[0],near[1]])
        if np.linalg.norm(np.array(new_node) - np.array(goal))<2:
            print("goal found, final node", new_node)
            break
    tree_list=np.array(tree)
    rrt_path=[goal]
    n=len(tree_list)-1
    x = tree_list[n,0]
    y = tree_list[n,1]
    f_x = tree_list[n,2]
    f_y = tree_list[n,3]
    rrt_path.append([x,y])
    search_list=[]
    while start not in rrt_path:
        search_list = tree_list[np.where((tree_list[:,0]==f_x) & (tree_list[:,1]==f_y))][0]
        search_list = search_list.tolist()
        rrt_path.append([search_list[0],search_list[1]])
        f_x = search_list[2]
        f_y = search_list[3]
    rrt_path=linear_interpolation(rrt_path)
    if to_torch:
        rrt_path = torch.tensor(rrt_path)
    return rrt_path 