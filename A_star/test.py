try:
    from queue import PriorityQueue
except ImportError:
    from Queue import PriorityQueue
import numpy as np
from enum import Enum

class Action(Enum):
    left = (0, -1, 1)
    right = (0, 1, 1)
    up = (-1, 0, 1)
    down = (1, 0, 1)

    def __str__(self):
        if self == self.left:
            return '<'
        elif self == self.right:
            return '>'
        elif self == self.up:
            return '^'
        elif self == self.down:
            return 'v'

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    valid = [Action.up, Action.left, Action.right, Action.down]
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if node off gride or obstacle
    if x - 1 < 0 or grid[x - 1, y] == 1:  #
        valid.remove(Action.up)  # romove corresponding element
    if x + 1 > n or grid[x + 1, y] == 1:
        valid.remove(Action.down)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid.remove(Action.left)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid.remove(Action.right)
    return valid


def visualize_path(grid, path, start):
    sgrid = np.zeros(np.shape(grid), dtype=np.str)
    sgrid[:] = ''
    sgrid[grid[:] == 1] = '0'
    pos = start

    for a in path:
        da = a.value
        sgrid[pos[0], pos[1]] = str(a)
        pos = (pos[0] + da[0], pos[1] + da[1])
    sgrid[pos[0], pos[1]] = 'G'
    sgrid[start[0], start[1]] = 'S'
    return sgrid

#heuristic function
def heuristic(position,goal_position):
    h = np.sqrt((position[0]-goal_position[0])**2+(position[1]-goal_position[1])**2)#euclidean
    return h

def a_star(grid,h,start,goal):
    path = []
    path_cost = 0
    queque = PriorityQueue()#set priority
    queque.put((0,start))#put priority+date
    visited = set(start)#not order not repeat
    branch = {}
    found =  False
    while not queque.empty():
        item = queque.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]
        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid,current_node):
                da = action.delta
                next_node = (current_node[0] + da[0],current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queque_cost = branch_cost +h(next_node,goal)
                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost,current_node,action)
                    queque.put((queque_cost,next_node))#put priority+data
    if found:
        n =goal
        path_cost = branch[n][0]
        while branch[n][1] != start:
            path.append(branch[n][2])
            n = branch[n][1]
        path.append(branch[n][2])
    else:
        print('***********')
        print('Failed To Find A Path!')
        print('***********')
    return path[::-1],path_cost


start = (0,0)
goal = (4,4)
grid = np.array([
    [0,1,0,0,0,0],
    [0,0,0,0,0,0],
    [0,1,0,0,0,0],
    [0,0,0,1,1,0],
    [0,0,0,1,0,0],
])

path,cost = a_star(grid,heuristic,start,goal)
print(path,cost)
visualize_path(grid,path,start)