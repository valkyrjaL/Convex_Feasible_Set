####################
# optimization.py
# this file implements the CFS algorithm
#
# Author: Ching-Ting Lin
# Copyright: 2019
####################


import math
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
solvers.options['show_progress'] = False

# First Order Model

def CFS(x0, x_ref, obstacles = [], cq = [1,1,1], cs = [1,1,1], theta = 0, 
minimal_dis = 0, ts = 1, maxIter = 10, SCCFS = False, slack_w = 1.0, stop_eps = 1e-3, 
trapezoid_orientation = [], xrec = []):
    
    n_ob = len(obstacles)
    if n_ob == 0:
        return x_ref
    x_rs = np.array(x_ref)
    h = x_rs.shape[0]    
    dimension = x_rs.shape[1] # would be 2 (x and y)
    x_rs = np.reshape(x_rs, (x_rs.size, 1)) # flatten to one dimension for applying qp, in the form of x0,y0,x1,y1,...
    x_rs = np.append(x_rs, [1]) # constant attribute
    if SCCFS is True: # append slack variables
        x_rs = np.append(x_rs, [0] * h * n_ob)
    x_origin = x_rs

    I = np.identity(h * dimension + 1)

    Velocity = np.zeros(((h - 1) * dimension, h * dimension + 1))
    for i in range(len(Velocity)):
        Velocity[i][i] = 1.0
        Velocity[i][i + dimension] = -1.0
    Velocity /= ts
    # print(Velocity)
    
    Acceleration = np.zeros(((h - 2) * dimension, h * dimension + 1))
    for i in range(len(Acceleration)):
        Acceleration[i][i] = 1.0
        Acceleration[i][i + dimension] = -2.0
        Acceleration[i][i + dimension + dimension] = 1.0
    Acceleration /= (ts * ts)
    # print(Acceleration)

    Q = cq[0] * I + cq[1] * np.dot(np.transpose(Velocity), Velocity) + cq[2] * np.dot(np.transpose(Acceleration), Acceleration)
    S = cs[0] * I + cs[1] * np.dot(np.transpose(Velocity), Velocity) + cs[2] * np.dot(np.transpose(Acceleration), Acceleration)
    if SCCFS is True:
        Q = np.hstack((Q, np.zeros((Q.shape[0], h * n_ob))))
        Q = np.vstack((Q, np.zeros((h * n_ob, len(x_rs)))))
        S = np.hstack((S, np.zeros((S.shape[0], h * n_ob))))
        S = np.vstack((S, np.zeros((h * n_ob, len(x_rs)))))

    C = np.zeros_like(Q)
    if len(xrec) >0 :
        C[0,0:4] = np.array([-2,0,1,0])
        C[1,0:4] = np.array([0,-2,0,1])
        C[0,h * dimension] = xrec[0]
        C[1,h * dimension] = xrec[1]
        C = np.dot(np.transpose(C), C)/(ts**4)
        Cf = np.zeros(Q.shape[0])
        Cf[:4] = np.array([-4*xrec[0], -4*xrec[1], 2*xrec[0], 2*xrec[1]])/(ts**4)

    w1 = 1
    w2 = 1
    w3 = 50
    H = w1 * Q + w2 * S + w3 * C
    f = -2 * w1 * np.dot(Q, x_origin)# + w3 * Cf
    b = np.ones((h * n_ob, 1)) * (-minimal_dis)
    if SCCFS is True:
        H[-h * n_ob:, -h * n_ob:] = np.identity(h * n_ob) * slack_w
        b = np.vstack((b, np.zeros((h * n_ob, 1))))
    H = matrix(H,(len(H),len(H[0])),'d')
    f = matrix(f,(len(f), 1),'d')
    b = matrix(b,(len(b),1),'d')

    J0 = w1 * np.dot(np.transpose(x_rs - x_origin), np.dot(Q, (x_rs - x_origin))) + w2 * np.dot(np.transpose(x_rs), np.dot(S, x_rs))
    J = float('inf')
    dlt = float('inf')
    cnt = 0

    while dlt > stop_eps:
        cnt += 1

        # Aeq * X = beq restrict x0 to be not changed.
        # '''
        Aeq = np.zeros((dimension + 1, len(x_rs)))
        Aeq[0][0] = 1
        Aeq[1][1] = 1
        Aeq[2][h * dimension] = 1
        beq = np.zeros((len(Aeq), 1))
        beq[0] = x0[0]
        beq[1] = x0[1]
        beq[2] = 1
        Aeq = matrix(Aeq,(len(Aeq),len(Aeq[0])),'d')
        beq = matrix(beq,(len(beq),1),'d')

        # Constraint * X <= b limits search space in a convex hull, i.e. feasible set.
        Constraint = np.zeros((h * n_ob, len(x_rs)))
        for i in range(h):
            x_r = x_rs[i * dimension : (i + 1) * dimension]
            line_set = convex_hull_2d_2_feasible_set(x_r, obstacles, i*ts, theta = theta, trapezoid_orientation = trapezoid_orientation)
            for j in range(n_ob):
                # line normal vector x, y
                x = line_set[j][0][0]
                y = line_set[j][0][1]
                const = line_set[j][1]
                Constraint[i * n_ob + j][i * dimension] = -x
                Constraint[i * n_ob + j][i * dimension + 1] = -y
                Constraint[i * n_ob + j][h * dimension] = const
                if SCCFS is True:
                    Constraint[i * n_ob + j][h * dimension + i * n_ob + j + 1] = -1.0

        if SCCFS is True:
            Constraint = np.vstack((Constraint, 
            np.hstack((np.zeros((h * n_ob, len(x_rs) - h * n_ob)), -np.identity(h * n_ob)))))
        Constraint = matrix(Constraint,(len(Constraint),len(Constraint[0])),'d')
        
        sol = solvers.qp(H, f, Constraint, b, Aeq, beq)
        x_ts = sol['x']
        x_ts = np.reshape(x_ts, len(x_rs))

        J = w1 * np.dot(np.transpose(x_ts - x_origin), np.dot(Q, (x_ts - x_origin))) + w2 * np.dot(np.transpose(x_ts), np.dot(S, x_ts))
        if SCCFS is True:
            J += np.linalg.norm(x_ts[-h * n_ob:]) **2 *slack_w
            J0 += np.linalg.norm(x_rs[-h * n_ob:]) **2 *slack_w
        dlt = min(abs(J - J0), np.linalg.norm(x_ts - x_rs))
        # print(abs(J - J0), " / ", np.linalg.norm(x_ts - x_rs))
        J0 = J
        x_rs = x_ts
        if cnt >= maxIter:
            break
    # print("cnt: ", cnt)
    x_rs = x_rs[: h * dimension]
    return x_rs.reshape(h, dimension)

def convex_hull_2d_2_feasible_set(x_r, obstacles = [], t = 0, obs_velocity = [0, 0], theta = 0, trapezoid_orientation = []):
    n_ob = len(obstacles)
    line_set = []
    for i in range(n_ob):

        dx = obs_velocity[0] * t
        dy = obs_velocity[1] * t

        # vehicle angle
        vh_l = 2.8 + 1.0
        vh_w = 1.2 + 0.6        
        a = vh_l / 2
        b = vh_w / 2

        # obstacle position and velocity, for the simulator
        if isinstance(obstacles[0][0], (list, np.ndarray)):
            [X,V] = obstacles[i]
            if theta == 0:
                v0 = [X[0] + a*V[0] + b*V[1], X[1] + a*V[1] - b*V[0]]   # upper right
                v1 = [X[0] - a*V[0] + b*V[1], X[1] - a*V[1] - b*V[0]]   # lower right
                v2 = [X[0] - a*V[0] - b*V[1], X[1] - a*V[1] + b*V[0]]   # lower left
                v3 = [X[0] + a*V[0] - b*V[1], X[1] + a*V[1] + b*V[0]]   # upper left
            else:   # outline the trapezoid
                d = np.tan(theta) * vh_w
                if trapezoid_orientation[i] == 0:   # at lane 0
                    v0 = [X[0] + a*V[0] + b*V[1], X[1] + a*V[1] - b*V[0]]
                    v1 = [X[0] - a*V[0] + b*V[1], X[1] - a*V[1] - b*V[0]]
                    v2 = [X[0] - a*V[0] - 3*vh_w*V[1], X[1] - a*V[1] + 3*vh_w*V[0]]
                    v3 = [X[0] + a*V[0] - 3*vh_w*V[1], X[1] + a*V[1] + 3*vh_w*V[0]]
                    v2 = [v2[0] - 3*vh_w*d*V[0], v2[1] - 3*vh_w*d*V[1]]
                    v3 = [v3[0] + 3*vh_w*d*V[0], v3[1] + 3*vh_w*d*V[1]]
                else:   # at other lane
                    v0 = [X[0] + a*V[0] + 3*vh_w*V[1], X[1] + a*V[1] - 3*vh_w*V[0]]
                    v1 = [X[0] - a*V[0] + 3*vh_w*V[1], X[1] - a*V[1] - 3*vh_w*V[0]]
                    v2 = [X[0] - a*V[0] - b*V[1], X[1] - a*V[1] + b*V[0]]
                    v3 = [X[0] + a*V[0] - b*V[1], X[1] + a*V[1] + b*V[0]]
                    v0 = [v0[0] + 3*vh_w*d*V[0], v0[1] + 3*vh_w*d*V[1]]
                    v1 = [v1[0] - 3*vh_w*d*V[0], v1[1] - 3*vh_w*d*V[1]]
        # obstacle position for tests
        else:
            obs = [obstacles[i][0] + dx, obstacles[i][1] + dy]
            v0 = [obs[0] - vh_w / 2, obs[1] + vh_l / 2]
            v1 = [obs[0] + vh_w / 2, obs[1] + vh_l / 2]
            v2 = [obs[0] + vh_w / 2, obs[1] - vh_l / 2]
            v3 = [obs[0] - vh_w / 2, obs[1] - vh_l / 2]
        v = [v0, v1, v2, v3]    # 4 vertices represent a surrounding vehicle
        normal, const, dist, land_point = distancePointMesh(x_r, v)

        line_set.append([normal, const])
    return line_set

def distancePointMesh(point, vertices):
# Input:
# point: a 2D point [x,y]
# vertices: an array of 2D points to represent a polygon
# 
# Output:
# ret_normal: a normalized normal vector points outward from the polygon
# ret_const: ret_normal * ret_land_point = ret_const
# ret_dist: min distance from the point to the line
# ret_land_point: a point on the line

    ret_dist = float('inf')
    ret_normal = []
    ret_land_point = []
    ret_const = 0
    ret_p = 0

    n_edge = len(vertices)
    for i in range(n_edge):
        x1 = vertices[i][0]
        y1 = vertices[i][1]
        x2 = vertices[(i + 1) % n_edge][0]
        y2 = vertices[(i + 1) % n_edge][1]

        trid = []
        trid += [np.linalg.norm([x1 - x2, y1 - y2])]
        trid += [np.linalg.norm([x1 - point[0], y1 - point[1]])]
        trid += [np.linalg.norm([x2 - point[0], y2 - point[1]])]

        # if the point is in between the line segment
        normal = [y1 - y2, x2 - x1]
        land_point = [x1, y1]
        const = -x1 * y2 + x2 * y1 # simply from normal * land_point'
        dist = abs(normal[0] * point[0] + normal[1] * point[1] - const) / trid[0]

        if trid[1]** 2 > trid[0]** 2 + trid[2]**2:
            dist = trid[2]
            normal = [point[0] - x2, point[1] - y2]
            const = np.dot(normal, np.transpose([x2, y2]))
            land_point = [x2, y2]

        if trid[2]** 2 > trid[0]**2 + trid[1]**2:
            dist = trid[1]
            normal = [point[0] - x1, point[1] - y1]
            const = np.dot(normal, np.transpose([x1, y1]))
        
        if dist < ret_dist:
            ret_dist = dist
            ret_normal = np.array(normal)
            ret_const = const
            ret_land_point = land_point
            ret_p = i

    # direction of normal vector, use a diagonal point to determine
    if np.dot(ret_normal, np.transpose(vertices[(ret_p + 2) % n_edge])) > ret_const:
        ret_normal = -ret_normal
        ret_const = -ret_const

    # normalization
    n = np.linalg.norm(ret_normal)
    ret_normal /= n
    ret_const /= n

    return ret_normal, ret_const, ret_dist, ret_land_point

'''
#print(getTrajectory([0,0],[[1,0],[2,0]],[[[1,0.5]]*2],2))
traj = CFS_FirstOrder([0,0],[[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[8,0],[9,0],[10,0]],[[[2,0.5]]*10,[[4,-0.5]]*10],10)
traj = np.array(matrix(traj).trans())
print(traj[0],traj[1])
fig1 = plt.figure()
plt.plot(traj[0],traj[1],'r-')
#plt.xlim(0, 6)
#plt.ylim(-1, 1)
plt.xlabel('x')
plt.title('test')
plt.show()
'''

if __name__ == "__main__":
    vh_w = 1.2
    vh_l = 2.8
    obstacles = [[0,3.0]]
    refTraj = np.array([[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,9]])

    plt.subplot(1,3,1)
    plt.title('CFS')
    traj1 = CFS(refTraj[0], refTraj, obstacles, cq = [0.1,0,0], cs = [0.1,0,1], minimal_dis = 0.1, maxIter = 10, SCCFS = False)
    traj1 = np.transpose(traj1)
    plt.plot(traj1[0],traj1[1],'g-*', label = "CFS")
    plt.plot(refTraj[:,0],refTraj[:,1],'k-*', label = "reference traj")

    for i in obstacles:
        rectangle = plt.Rectangle((i[0] - vh_w/2, i[1] - vh_l/2), vh_w, vh_l, fc='k', alpha = 0.2)
        plt.gca().add_patch(rectangle)
    plt.axis('scaled')
    plt.ylim((0,10))
    plt.xlim((-1.0,1.5))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))

    plt.subplot(1,3,2)
    plt.title('CFS vs SCCFS')
    plt.plot(traj1[0],traj1[1],'g-*', label = "CFS")

    traj2 = CFS(refTraj[0], refTraj, obstacles, cq = [0.1,0,0], cs = [0.1,0,1], minimal_dis = 0.1, maxIter = 10, SCCFS = True, slack_w = 1.0)
    traj2 = np.transpose(traj2)
    plt.plot(traj2[0],traj2[1],'b-*', label = "SCCFS")

    for i in obstacles:
        rectangle = plt.Rectangle((i[0] - vh_w/2, i[1] - vh_l/2), vh_w, vh_l, fc='k', alpha = 0.2)
        plt.gca().add_patch(rectangle)
    plt.axis('scaled')
    plt.ylim((0,10))
    plt.xlim((-1.0,1.5))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))

    plt.subplot(1,3,3)
    plt.title('SCCFS')
    plt.plot(traj2[0],traj2[1],'b-*', label = "slack_w 1.0")
    traj3 = CFS(refTraj[0], refTraj, obstacles, cq = [0.1,0,0], cs = [0.1,0,10], minimal_dis = 0.1, maxIter = 10, SCCFS = True, slack_w = 1.0)
    traj3 = np.transpose(traj3)
    plt.plot(traj3[0],traj3[1],'m-*', label = "slack_w 2.0")

    for i in obstacles:
        rectangle = plt.Rectangle((i[0] - vh_w/2, i[1] - vh_l/2), vh_w, vh_l, fc='k', alpha = 0.2)
        plt.gca().add_patch(rectangle)
    plt.axis('scaled')
    plt.ylim((0,10))
    plt.xlim((-1.0,1.5))    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))

    plt.show()