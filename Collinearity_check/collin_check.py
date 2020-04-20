#you can check any three points collinearity by a matrix
import numpy as np
#define three points
p1 = np.array([1, 2])
p2 = np.array([2, 3])
p3 = np.array([3, 4])

#introduce threshold to deal with numerical precision issues
# and/or allow a tolerance for collinearity

#3D
def point(p):
    return np.array([p[0],p[1],1.])

def collinearity_float(p1,p2,p3,epsilon = 1e-6):
    collinear = False
    #stack the array by line
    mat = np.vstack((point(p1),point(p2),point(p3)))
    #calculate the determinant of the matrix
    det = np.linalg.det(mat)
    if det < epsilon:
        collinear = True
    return collinear

# 2D
# use 𝑑𝑒𝑡=𝑥1(𝑦2−𝑦3)+𝑥2(𝑦3−𝑦1)+𝑥3(𝑦1−𝑦2) to test
def collinearity_int(p1,p2,p3):
    collinear = False
    det = p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])
    if det == 0:
        collinear = True
    return collinear

#test
import time
t1 = time.time()
collinear_3d = collinearity_float(p1,p2,p3)
t_3D = time.time()-t1
print(collinear_3d)

t2 = time.time()
collinear_2d = collinearity_int(p1,p2,p3)
t_2D = time.time() - t2
print(collinear_2d)
#because the time is too short, it doesn't work sometimes
#print(t_3D/t_2D)