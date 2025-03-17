#!/usr/bin/env python3

import cv2
import yaml
import numpy as np
np.seterr(all="ignore")
import math
from geometry_msgs.msg import TransformStamped, Transform, PointStamped, PoseStamped, Pose

def load_tf_data(tf_data_path):
    with open(tf_data_path, 'r') as file:
        tf_data = yaml.safe_load(file)
    return tf_data

def vec2hat(x):
    if len(x.shape) == 2:
        return np.array([[  0,     -x[2][0],   x[1][0]],
                        [ x[2][0],     0,     -x[0][0]],
                        [-x[1][0],  x[0][0],   0]])
    elif len(x.shape) == 1:
        return np.array([[0, -x[2], x[1]],
                        [x[2], 0, -x[0]],
                        [-x[1], x[0], 0]])

def adjoint(g):
    out = np.zeros((6,6))
    out[:3,:3] = g[:3,:3]
    out[3:,:3] = vec2hat(g[:3,3]).dot(g[:3,:3])
    out[3:,3:] = g[:3,:3]
    return out

def exp2hat(xi):
    T = np.zeros((4,4))
    v = xi[0:3].reshape(-1,1)
    w = xi[3:6].reshape(-1,1)
    w_hat = vec2hat(w)
    T[:3,:3] = w_hat
    T[:3,3]  = v
    T[3,3]   = 1
    return T

def vecs2tfstamped(rvec,tvec,t,frame_id="",child_frame_id=""):
    msg = TransformStamped()
    msg.child_frame_id = child_frame_id
    msg.header.stamp = t
    msg.header.frame_id = frame_id
    msg.transform = vecs2tf(rvec,tvec)
    return msg

def vecs2posestamped(rvec,tvec,t,frame_id=""):
    msg = PoseStamped()
    msg.header.stamp = t
    msg.header.frame_id = frame_id
    msg.pose = vecs2pose(rvec,tvec)
    return msg

def gdotp(g,p):
    return g.dot(np.append(p, [1]))[:3]

def gdotv(g,v):
    return g.dot(np.append(v, [0]))[:3]

def g2tf(g):
    rvec = cv2.Rodrigues(g[:3,:3])[0]
    tvec = g[:3,3]
    return vecs2tf(rvec,tvec)

def vecs2tf(rvec, tvec):
    out = Transform()
    quat = np.float64(rvec2quat(rvec))
    out.rotation.x = quat[0][0]
    out.rotation.y = quat[1][0]
    out.rotation.z = quat[2][0]
    out.rotation.w = quat[3][0]
    out.translation.x = tvec[0]
    out.translation.y = tvec[1]
    out.translation.z = tvec[2]
    return out

def vecs2pose(rvec,tvec):
    out = Pose()
    quat = rvec2quat(rvec)
    out.orientation.x = quat[0]
    out.orientation.y = quat[1]
    out.orientation.z = quat[2]
    out.orientation.w = quat[3]
    out.position.x = tvec[0]
    out.position.y = tvec[1]
    out.position.z = tvec[2]
    return out

def transmsg2vecs(msg):
    tvec = np.zeros((1,3))
    tvec[0][0] = msg.translation.x
    tvec[0][1] = msg.translation.y
    tvec[0][2] = msg.translation.z
    quat = np.zeros((4,1))
    quat[0] = msg.rotation.x
    quat[1] = msg.rotation.y
    quat[2] = msg.rotation.z
    quat[3] = msg.rotation.w
    return tvec, quat

def pose2vecs(msg):
    tvec = np.zeros((1,3))
    tvec[0][0] = msg.position.x
    tvec[0][1] = msg.position.y
    tvec[0][2] = msg.position.z
    quat = np.zeros((4,1))
    quat[0] = msg.orientation.x
    quat[1] = msg.orientation.y
    quat[2] = msg.orientation.z
    quat[3] = msg.orientation.w
    return tvec, quat

def g2tfstamped(g,t,frame_id="",child_frame_id=""):
    rvec = cv2.Rodrigues(g[:3,:3])[0]
    tvec = g[:3,3]
    return vecs2tfstamped(rvec,tvec,t,frame_id,child_frame_id)

def g2posestamped(g,t,frame_id=""):
    rvec = cv2.Rodrigues(g[:3,:3])[0]
    tvec = g[:3,3]
    return vecs2posestamped(rvec,tvec,t,frame_id)

def tfstamped2g(msg):
    tvec, quat = transmsg2vecs(msg.transform)
    rvec = quat2rvec(quat)
    g = cv2vecs2g(rvec,tvec)
    return g

def tfstamped2ginv(msg):
    tvec, quat = transmsg2vecs(msg.transform)
    rvec = quat2rvec(quat)
    ginv = cv2vecs2ginv(rvec,tvec)
    return ginv

def posestamped2g(msg):
    tvec, quat = pose2vecs(msg.pose)
    rvec = quat2rvec(quat)
    g = cv2vecs2g(rvec,tvec)
    return g

def posestamped2ginv(msg):
    tvec, quat = pose2vecs(msg.pose)
    rvec = quat2rvec(quat)
    ginv = cv2vecs2ginv(rvec,tvec)
    return ginv

def gen_g(R,p):
    T = np.zeros((4,4))
    T[:3,:3] = R
    T[:3,3]  = p.ravel()
    T[3,3]   = 1
    return T

def cv2vecs2g(rvec, tvec):
    # T = np.vstack((np.hstack((w_hat,v)),np.array([0,0,0,0])))
    R = cv2.Rodrigues(rvec)[0]
    p = tvec.reshape(-1,1)
    g = gen_g(R,p)
    return g

def cv2vecs2ginv(rvec,tvec):
    R = cv2.Rodrigues(rvec)[0]
    p = tvec.reshape(-1,1)
    ginv = gen_g(R.T,-R.T.dot(p))
    return ginv

def ginv(g):
    R = g[:3,:3]
    p = g[:3,3].reshape(-1,1)
    return gen_g(R.T,-R.T.dot(p))

def g2twist(g):
    R = g[:3,:3]
    p = g[:3,3]
    rot_exp_coord = cv2.Rodrigues(R)[0]
    th = np.linalg.norm(rot_exp_coord)
    w = rot_exp_coord/th
    v = np.linalg.inv((np.identity(3)-R).dot(vec2hat(w))+w.dot(w.T)*th).dot(p).reshape(-1,1)
    return v, w, th

def twist2g(xi,th):
    w = xi[3:6].reshape(-1,1)
    v = xi[:3].reshape(-1,1)
    R = cv2.Rodrigues(w*th)[0]
    p = (np.eye(3)-R).dot(np.cross(w,v))+w.dot(w.T).dot(v)*th
    g = gen_g(R,p)
    return g
    
def g2quat(g):
    R = g[:3,:3]
    rvec = cv2.Rodrigues(R)[0]
    theta = np.linalg.norm(rvec)
    w = rvec/theta
    quat = np.zeros((4,1))
    quat[:3] = w*math.sin(theta/2)
    quat[3] = math.cos(theta/2)
    return quat

def quat2rvec(quat):
    w = np.zeros((3,1))
    theta = math.acos(quat[3])*2
    w = quat[:3]/math.sqrt(1-quat[3]**2)
    return w*theta

def rvec2quat(rvec):
    theta = np.linalg.norm(rvec)
    if theta < 1e-9:
        w = rvec
    else:
        w = rvec/theta
    quat = np.zeros((4,1))
    quat[:3] = w*math.sin(theta/2)
    quat[3] = math.cos(theta/2)
    return quat

def twist2screw(v,w,th):
    q = np.cross(w.reshape(-1),v.reshape(-1)).reshape(-1,1)
    h = w.T.dot(v)
    u = w
    M = th
    return q, h, u, M

def distance(xdiff,zdiff):
    dist = math.sqrt(xdiff**2 + zdiff**2)
    return dist

def rot_dist(rvec1, rvec2):
    R1 = cv2.Rodrigues(rvec1)[0]
    R2 = cv2.Rodrigues(rvec2)[0]
    return math.acos((np.trace(R1.dot(R2.T))-1)/2)

def psm_g(z1,z2,z3,z4,z5,z6,z7,th1,th2,th3,th4,th5,th6,th7):
    g1 = twist2g(z1,th1)
    g2 = twist2g(z2,th2)
    g3 = twist2g(z3,th3)
    g4 = twist2g(z4,th4)
    g5 = twist2g(z5,th5)
    g6 = twist2g(z6,th6)
    g7 = twist2g(z7,th7)
    g = g1.dot(g2.dot(g3.dot(g4.dot(g5.dot(g6.dot(g7))))))
    return g

def pt3d2ptstamped(pt, ts, frame_id=""):
    if pt is None:
        return None
    pt = np.float64(pt)
    msg = PointStamped()
    msg.header.stamp = ts
    msg.header.frame_id = frame_id
    msg.point.x = pt[0]
    msg.point.y = pt[1]
    msg.point.z = pt[2]
    return msg

def ptstamped2pt3d(msg):
    pt = np.zeros((3,))
    pt[0] = msg.point.x
    pt[1] = msg.point.y
    pt[2] = msg.point.z
    return pt

# g_psm1tip_psm1jaw = cv2vecs2g(
#     np.array([0.0,0.0,0.0]), np.array([-0.005, -0.0025, 0.0147])
# )

g_psm1tip_psm1jaw = cv2vecs2g(
    np.array([0.0,0.0,0.0]), np.array([-0.003, -0.002, 0.015])
)

g_pchjaw_offset = cv2vecs2g(
    math.radians(-10)*np.array([0.0,0.0,1.0]), np.array([0,0,0])
)

g_psm2tip_psm2jaw = cv2vecs2g(
    math.radians(-10)*np.array([1.0,0.0,0.0]), np.array([-0.004, 0.002, 0.018])
)

g_map_odom = cv2vecs2g(np.array([0,0,1])*math.radians(90),np.array([0,0,1])).dot(
    cv2vecs2g(np.array([1,0,0])*math.radians(90),np.array([0,0,0]))
)

def g_ecm_dvrk(cam_type):
    if cam_type == "30":
        return cv2vecs2g(np.array([1,0,0])*math.radians(30),np.array([0,0,0]))
    elif cam_type == "0":
        return cv2vecs2g(np.array([0.0,0.0,0.0]),np.array([0,0,0]))

def tf_dist(T1, T2, wT=1, wR=0.1):
    """
    Computes the total distance between two homogeneous transformation matrices.

    Args:
        T1 (np.ndarray): First 4x4 transformation matrix.
        T2 (np.ndarray): Second 4x4 transformation matrix.
        wT (float): Translational distance weight.
        wR (float): Rotational distance weight.

    Returns:
        float: Total transformation distance (weighted sum of rotation and translation).
        float: Rotation distance in radians.
        float: Translation distance in meters.
    """
    # Extract rotation matrices and translation vectors
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]

    # Compute translation distance (Euclidean norm)
    trans_dist = np.linalg.norm(t1 - t2)

    # Compute rotation distance using acos(trace formula)
    rot_dist = np.clip((np.trace(R1.dot(R2.T)) - 1) / 2, -1.0, 1.0)  # Clip for numerical stability
    rot_dist = math.acos(rot_dist)

    # Combine both distances (you can adjust the weighting factor if needed)
    total_dist = wT*trans_dist + wR*rot_dist

    return total_dist, trans_dist, rot_dist