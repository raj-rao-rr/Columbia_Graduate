import numpy as numpy
from queue import PriorityQueue
from utils.utils import PathPlanMode, Heuristic, cost, expand, visualize_expanded, visualize_path
import numpy as np


def uninformed_search(grid, start, goal, mode: PathPlanMode):
    """ Find a path from start to goal in the gridworld using 
    BFS or DFS.
    
    Args:
        grid (numpy): NxN numpy array representing the world,
        with terrain features encoded as integers.
        start (tuple): The starting cell of the path.
        goal (tuple): The ending cell of the path.
        mode (PathPlanMode): The search strategy to use. Must
        specify either PathPlanMode.DFS or PathPlanMode.BFS.
    
    Returns:
        path (list): A list of cells from start to goal.
        expanded (list): A list of expanded cells.
        frontier_size (list): A list of integers containing
        the size of the frontier at each iteration.
    """

    frontier = [start]
    frontier_sizes = []
    expanded = []
    reached = {start: None}
    
    # TODO:    
        
    path = []
    
    if mode == 1:
        
        # performing DFS 
        while len(frontier) > 0:
            
            frontier_sizes.append(len(frontier))
            node = frontier.pop() # LIFO
            expanded.append(node)
            
            if node == goal:
                print(f'We reached goal')
                
                # work backwards to derive the path sequence
                t = goal
                while t != start:                
                    path.append(reached[t])
                    t = reached[t]
                
                return path[::-1], expanded, frontier_sizes
            
            # assign child node to parent node during expansion
            for child in expand(grid, node):
                if child not in reached:
                    reached[child] = node
                    frontier.append(child)
        
        return path, expanded, frontier_sizes
    
    if mode == 2:
        
        # performing BFS
        while len(frontier) > 0:
            
            frontier_sizes.append(len(frontier))
            node = frontier.pop(0) # FIFO
            expanded.append(node)
            
            if node == goal:
                print(f'We reached goal')
                
                # work backwards to derive the path sequence
                t = goal
                while t != start:                
                    path.append(reached[t])
                    t = reached[t]
                
                return path[::-1], expanded, frontier_sizes
            
            # assign child node to parent node during expansion
            for child in expand(grid, node):
                if child not in reached:
                    reached[child] = node
                    frontier.append(child)
        
        return path, expanded, frontier_sizes


def h_func(current_node, end_node, h: int):
    """Peform a heuristic function construction based on distance between
    nodes in the given search grid
    """
    
    if h == 1: # compute the L1 norm (Manhattan)
        return abs(end_node[0] - current_node[0]) + abs(end_node[1] - current_node[1])    
    elif h == 2: # compute the L2 norm (Ecluedian)
        return np.sqrt( (end_node[0] - current_node[0])**2 + (end_node[1] - current_node[1])**2 )
    else:
        pass
        
    
def a_star(grid, start, goal, mode: PathPlanMode, heuristic: Heuristic, width):
    """ Performs A* search or beam search to find the
    shortest path from start to goal in the gridworld.
    
    Args:
        grid (numpy): NxN numpy array representing the world,
        with terrain features encoded as integers.
        start (tuple): The starting cell of the path.
        goal (tuple): The ending cell of the path.
        mode (PathPlanMode): The search strategy to use. Must
        specify either PathPlanMode.A_STAR or
        PathPlanMode.BEAM_SEARCH.
        heuristic (Heuristic): The heuristic to use. Must
        specify either Heuristic.MANHATTAN or Heuristic.EUCLIDEAN.
        width (int): The width of the beam search. This should
        only be used if mode is PathPlanMode.BEAM_SEARCH.
    
    Returns:
        path (list): A list of cells from start to goal.
        expanded (list): A list of expanded cells.
        frontier_size (list): A list of integers containing
        the size of the frontier at each iteration.
    """

    frontier = PriorityQueue()
    frontier.put((0, start))
    frontier_sizes = []
    expanded = []
    reached = {start: {"cost": cost(grid, start), "parent": None}}

    # TODO:

    path = []
    
    if mode == 3:
        
        # perform A* Search
        while frontier.qsize() > 0:
            
            frontier_sizes.append(frontier.qsize())
            current_cost, node = frontier.get()
            expanded.append(node)
            
            if node == goal:
                print(f'We reached goal')
                
                # work backwards to derive the path sequence
                t = goal
                while t != start:                
                    path.append(reached[t]['parent'])
                    t = reached[t]['parent']
                    
                return path[::-1], expanded, frontier_sizes
            
            # assign child node to parent node during expansion
            for child in expand(grid, node):
                state_cost = cost(grid, node) + h_func(child, goal, heuristic)
                
                # check to see if the new cost is lower than its old one.
                if child not in reached or state_cost < reached[child]['cost']:
                    reached[child] = {"cost": cost(grid, child), 
                                      "parent": node}
                    frontier.put( (state_cost, child) )
        
        return path, expanded, frontier_sizes
    
    if mode == 4:
        
        # perform A* Search
        while frontier.qsize() > 0:
            
            frontier_sizes.append(frontier.qsize())
            current_cost, node = frontier.get()
            expanded.append(node)
            
            if node == goal:
                print(f'We reached goal')
                
                # work backwards to derive the path sequence
                t = goal
                while t != start:                
                    path.append(reached[t]['parent'])
                    t = reached[t]['parent']
                
                return path[::-1], expanded, frontier_sizes
            
            # assign child node to parent node during expansion
            for child in expand(grid, node):
                state_cost = cost(grid, node) + h_func(child, goal, heuristic)
                
                if child not in reached or state_cost < reached[child]['cost']:
                    reached[child] = {"cost": cost(grid, child), 
                                      "parent": node}
                    frontier.put( (state_cost, child) )
            
            # width truncation of the frontier list
            temp_que = PriorityQueue()
            selection = width if frontier.qsize() > width else frontier.qsize()
            for _ in range(selection):
                temp_que.put(frontier.get())
            frontier = temp_que
            
        return path, expanded, frontier_sizes


def local_search(grid, start, goal, heuristic: Heuristic):
    """ Find a path from start to goal in the gridworld using
    local search.

    Args:
        grid (numpy): NxN numpy array representing the world,
        with terrain features encoded as integers.
        start (tuple): The starting cell of the path.
        goal (tuple): The ending cell of the path.
        heuristic (Heuristic): The heuristic to use. Must
        specify either Heuristic.MANHATTAN or Heuristic.EUCLIDEAN.

    Returns:
        path (list): A list of cells from start to goal.
    """
    
    max_itter = round(1e4)
    count = 0
    path = [start]
    
    # TODO
    current = start
    while count < max_itter: 
        succesor_states = expand(grid, current)
        
        # check all adjacent nodes h-values
        lowest_node = (None, None)
        lowest_value = np.inf
        for node in succesor_states:
            
            h_val = h_func(node, goal, heuristic)
            if h_val < lowest_value: 
                lowest_node = node
                lowest_value = h_val
        
        current = lowest_node
        path.append(current)
        
        if current == goal:
            print('Goal Found')
            return path
            
        count += 1
        
    if count >= max_itter:
        print('Reached max-iterations, no solution found')
        path = []
    
    return path


def test_world(world_id, start, goal, h, width, animate, world_dir):
    print(f"Testing world {world_id}")
    grid = np.load(f"{world_dir}/world_{world_id}.npy")

    if h == 0:
        modes = [
            PathPlanMode.DFS,
            PathPlanMode.BFS
        ]
        print("Modes: 1. DFS, 2. BFS")
    elif h == 1 or h == 2:
        modes = [
            PathPlanMode.A_STAR,
            PathPlanMode.BEAM_SEARCH
        ]
        if h == 1:
            print("Modes: 1. A_STAR, 2. BEAM_A_STAR")
            print("Using Manhattan heuristic")
        else:
            print("Modes: 1. A_STAR, 2. BEAM_A_STAR")
            print("Using Euclidean heuristic")
    elif h == 3 or h == 4:
        h -= 2
        modes = [
            PathPlanMode.LOCAL_SEARCH
        ]
        if h == 1:
            print("Mode: LOCAL_SEARCH")
            print("Using Manhattan heuristic")
        else:
            print("Mode: LOCAL_SEARCH")
            print("Using Euclidean heuristic")

    for mode in modes:
        
        search_type, path, expanded, frontier_size = None, [], [], []
        if mode == PathPlanMode.DFS:
            path, expanded, frontier_size = uninformed_search(grid, start, goal, mode)
            search_type = "DFS"
        elif mode == PathPlanMode.BFS:
            path, expanded, frontier_size = uninformed_search(grid, start, goal, mode)
            search_type = "BFS"
        elif mode == PathPlanMode.A_STAR:
            path, expanded, frontier_size = a_star(grid, start, goal, mode, h, 0)
            search_type = "A_STAR"
        elif mode == PathPlanMode.BEAM_SEARCH:
            path, expanded, frontier_size = a_star(grid, start, goal, mode, h, width)
            search_type = "BEAM_A_STAR"
        elif mode == PathPlanMode.LOCAL_SEARCH:
            path = local_search(grid, start, goal, h)
            search_type = "LOCAL_SEARCH"

        if search_type:
            print(f"Mode: {search_type}")
            path_cost = 0
            
            for c in path:
                path_cost += cost(grid, c)
            print(f"Path length: {len(path)}")
            print(f"Path cost: {path_cost}")
            if frontier_size:
                print(f"Number of expanded states: {len(frontier_size)}")
                print(f"Max frontier size: {max(frontier_size)}\n")
            if animate == 0 or animate == 1:
                visualize_expanded(grid, start, goal, expanded, path, animation=animate)
            else:
                visualize_path(grid, start, goal, path)