import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 12,12

#limit the steering angle range
STEERING_ANGLE_MAX = np.deg2rad(50)
def sample_steering_angle():
    return np.random.uniform(-STEERING_ANGLE_MAX,STEERING_ANGLE_MAX)
# you will define the Dubins's car model gy differental equations
def simulate(state,angle,v,dt):
    x = state[0]
    y = state[1]
    theta = state[2]
    new_x = x+v*np.cos(theta)*dt
    new_y = y+v*np.sin(theta)*dt
    new_theta = theta+v*np.tan(angle)*dt
    return [new_x,new_y,new_theta]

v = 2
dt = 0.1
total_time = 50
#initial state
states = [[0,0,0]]
for _ in np.arange(0,total_time,dt):
    angle = sample_steering_angle()
    state = simulate(states[-1],angle,v,dt)
    states.append(state)
states = np.array(states)

plt.plot(states[:,0],states[:,1],color='blue')
plt.axis('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()