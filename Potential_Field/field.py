import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 12,12

#attraction
def attraction(position,goal,alpha):
    return alpha * (np.array(position) - np.array(goal))

#repulsion
def repulsion(position,obstacle,beta,q_max):
    position = np.array(position)
    obstacle = np.array(obstacle)
    return beta * ((1/q_max) - (1/np.linalg.norm(position-obstacle)))*(1/np.linalg.norm(position-obstacle)**2)

#generate the potential field
def potential_field(grid,goal,alpha,beta,q_max):
    x =[]
    y =[]
    fx = []
    fy = []
    obs_i,obs_j = np.where(grid == 1)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i,j] == 0:
                #add attaraction force
                force = attraction([i,j],goal,alpha)
                for(oi,oj) in zip(obs_i,obs_j):
                    if np.linalg.norm(np.array([i,j])-np.array([oi,oj])) < q_max:
                        #add repulsion force
                        force += repulsion([i,j],[oi,oj],beta,q_max)

                x.append(i)
                y.append(j)
                fx.append(force[0])
                fy.append(force[1])
    return x,y,fx,fy

#generate environment
grid = np.zeros((30,30))
grid[10:15,10:15] = 1.0
grid[17:25,10:17] = 1.0
goal = [5,5]
#constants
alpha = 1.0
beta = 2.0
q_max = 10

x,y,fx,fy = potential_field(grid,goal,alpha,beta,q_max)
plt.imshow(grid, cmap = 'Greys', origin='lower')
plt.plot(goal[1], goal[0], 'ro')
plt.quiver(y,x,fy,fx)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


