#!/usr/bin/env python3

import cv2
import numpy as np
np.seterr(all="ignore")
import math
import scipy.io as sio
from scipy.optimize import least_squares, minimize
from utils import tf_utils
import time
from pyquaternion import Quaternion


def getTransformMatrix(xi,theta,k,m):

    eps = 1e-6
    w   = xi[3:6].reshape(-1,1)
    v   = xi[0:3].reshape(-1,1)
    th = k*theta + m
    w_norm = np.linalg.norm(w)

    if w_norm < eps:
        v /= np.linalg.norm(v)
        g  = tf_utils.gen_g(np.eye(3),v*th)
    else:
        w  /= w_norm
        v  /= w_norm
        R  = cv2.Rodrigues(w*th)[0]
        p  = (np.eye(3)-R).dot(np.cross(w,v,axis=0))+w.dot(w.T).dot(v)*th
        g  = tf_utils.gen_g(R,p)
        
    return g

def get_tip_pose(thetas, arm_calib_data):
    if "PSM" in arm_calib_data.arm:
        # Create the transformation matrices for the respective joints
        g1 = getTransformMatrix(arm_calib_data.xi1, thetas[0], arm_calib_data.k[0], arm_calib_data.m[0])
        g2 = getTransformMatrix(arm_calib_data.xi2, thetas[1], arm_calib_data.k[1], arm_calib_data.m[1])
        g3 = getTransformMatrix(arm_calib_data.xi3, thetas[2], arm_calib_data.k[2], arm_calib_data.m[2])
        g4 = getTransformMatrix(arm_calib_data.xi4, thetas[3], arm_calib_data.k[3], arm_calib_data.m[3])
        g5 = getTransformMatrix(arm_calib_data.xi5, thetas[4], arm_calib_data.k[4], arm_calib_data.m[4])
        g6 = getTransformMatrix(arm_calib_data.xi6, thetas[5], arm_calib_data.k[5], arm_calib_data.m[5])

        # Get the overall transformation matrix
        return g1.dot(g2).dot(g3).dot(g4).dot(g5).dot(g6).dot(arm_calib_data.gst0)

    elif arm_calib_data.arm == "ECM":
        # Create the transformation matrices for the respective joints
        g1 = getTransformMatrix(arm_calib_data.xi1, thetas[0], arm_calib_data.k[0], arm_calib_data.m[0])
        g2 = getTransformMatrix(arm_calib_data.xi2, thetas[1], arm_calib_data.k[1], arm_calib_data.m[1])
        g3 = getTransformMatrix(arm_calib_data.xi3, thetas[2], arm_calib_data.k[2], arm_calib_data.m[2])
        g4 = getTransformMatrix(arm_calib_data.xi4, thetas[3], arm_calib_data.k[3], arm_calib_data.m[3])

        # Get the overall transformation matrix
        return g1.dot(g2).dot(g3).dot(g4).dot(arm_calib_data.gst0)
    
def get_tip_pose_jaw(thetas, arm_calib_data):
    # Get the overall transformation matrix
    return get_tip_pose(thetas, arm_calib_data).dot(arm_calib_data.g_psmtip_psmjaw)

class get_arm_calib_data(object):
    def __init__(self,calib_fn):
        calib_data = sio.loadmat(calib_fn,squeeze_me=True)
        if "psm" in calib_fn:
            if "psm1" in calib_fn:
                psm_x      = calib_data["psm1_x"]
                self.arm   = "PSM1"
                self.gst0  = calib_data["psm1_gst0"].reshape(4,4)
                self.g_psmtip_psmjaw = tf_utils.cv2vecs2g(
                    np.array([0.0,0.0,0.0]),np.array([-0.005,-0.0025,0.0147])
                )
            elif "psm2" in calib_fn:
                psm_x      = calib_data["psm2_x"]
                self.arm   = "PSM2"
                self.gst0  = calib_data["psm2_gst0"].reshape(4,4).dot(
                    tf_utils.cv2vecs2g(
                        np.array([1.0, 0.0, 0.0])*np.radians(5), np.array([0.0, 0.0, 0.0])
                    )
                )
                self.g_psmtip_psmjaw = tf_utils.cv2vecs2g(
                    np.array([0.0,0.0,0.0]),np.array([-0.004, 0.0, 0.019])
                )
            self.xi1   = psm_x[0:6]
            self.xi2   = psm_x[6:12]
            self.xi3   = psm_x[12:18]
            self.xi4   = psm_x[18:24]
            self.xi5   = psm_x[24:30]
            self.xi6   = psm_x[30:36]
            self.k     = psm_x[36:42]
            self.m     = psm_x[42:48]
            
        elif "ecm" in calib_fn:
            ecm_x      = calib_data["ecm_x"]
            self.arm   = "ECM"
            self.gst0  = calib_data["ecm_gst0"].reshape(4,4)
            self.xi1   = ecm_x[0:6]
            self.xi2   = ecm_x[6:12]
            self.xi3   = ecm_x[12:18]
            self.xi4   = ecm_x[18:24]
            self.k     = ecm_x[24:28]
            self.m     = ecm_x[28:32]
        else:
            print("Invalid Arm Calibration File...")
            
class dvrk_custom_ik():
    def __init__(self, calib_fn, wT, wR, init_jp, Joffsets):
        self.arm_calib_data = get_arm_calib_data(calib_fn)
        # self.arm_fk = get_arm_fk(params["calib_fn"])
        self.wT = wT
        self.wR = wR
        self.target = np.eye(4)
        # self.constraints = self.load_constraints(params)
        self.constraints = self.get_constraints(init_jp, Joffsets)
        
    def __del__(self):
        print("Destructing class dvrk_custom_ik...")
        
    def load_constraints(self, params):
        # When using least_squares
        # return (
        #     np.array(
        #         [
        #             math.radians(params["joint1_min"]),
        #             math.radians(params["joint2_min"]),
        #             params["joint3_min"],
        #             math.radians(params["joint4_min"]),
        #             math.radians(params["joint5_min"]),
        #             math.radians(params["joint6_min"])
        #         ]
        #     ),
        #     np.array(
        #         [
        #             math.radians(params["joint1_max"]),
        #             math.radians(params["joint2_max"]),
        #             params["joint3_max"],
        #             math.radians(params["joint4_max"]),
        #             math.radians(params["joint5_max"]),
        #             math.radians(params["joint6_max"])
        #         ]
        #     )
        # )
        # When using minimize
        return (
            (math.radians(params["joint1_min"]),math.radians(params["joint1_max"])),
            (math.radians(params["joint2_min"]),math.radians(params["joint2_max"])),
            (params["joint3_min"],params["joint3_max"]),
            (math.radians(params["joint4_min"]),math.radians(params["joint4_max"])),
            (math.radians(params["joint5_min"]),math.radians(params["joint5_max"])),
            (math.radians(params["joint6_min"]),math.radians(params["joint6_max"]))
        )
    
    def get_constraints(self, init_jp, Joffsets):
        constraints = []
        for i, (jp, Joffset) in enumerate(zip(init_jp, Joffsets)):
            min_jp = jp - math.radians(Joffset)
            max_jp = jp + math.radians(Joffset)
            if i == 0:
                if min_jp < math.radians(-85):
                    min_jp = math.radians(-85)
                if max_jp > math.radians(85):
                    max_jp = math.radians(85)
            elif i == 1:
                if min_jp < math.radians(-45):
                    min_jp = math.radians(-45)
                if max_jp > math.radians(35):
                    max_jp = math.radians(35)
            elif i == 2:
                if min_jp < 0.01:
                    min_jp = 0.01
                if max_jp > 0.24:
                    max_jp = 0.24
            elif i == 3:
                if min_jp < math.radians(-160):
                    min_jp = math.radians(-160)
                if max_jp > math.radians(160):
                    max_jp = math.radians(160)
            elif i == 4:
                if min_jp < math.radians(-60):
                    min_jp = math.radians(-60)
                if max_jp > math.radians(60):
                    max_jp = math.radians(60)
            elif i == 5:
                if min_jp < math.radians(-95):
                    min_jp = math.radians(-95)
                if max_jp > math.radians(95):
                    max_jp = math.radians(95)
            constraints.append((min_jp, max_jp))
        return tuple(constraints)
        
    def jacobian(self, thetas):
        J = np.zeros((6,6))
        g1 = getTransformMatrix(self.arm_calib_data.xi1, thetas[0], self.arm_calib_data.k[0], self.arm_calib_data.m[0])
        g2 = getTransformMatrix(self.arm_calib_data.xi2, thetas[1], self.arm_calib_data.k[1], self.arm_calib_data.m[1])
        g3 = getTransformMatrix(self.arm_calib_data.xi3, thetas[2], self.arm_calib_data.k[2], self.arm_calib_data.m[2])
        g4 = getTransformMatrix(self.arm_calib_data.xi4, thetas[3], self.arm_calib_data.k[3], self.arm_calib_data.m[3])
        g5 = getTransformMatrix(self.arm_calib_data.xi5, thetas[4], self.arm_calib_data.k[4], self.arm_calib_data.m[4])
        J[:,0] = self.arm_calib_data.xi1
        J[:,1] = tf_utils.adjoint(g1).dot(self.arm_calib_data.xi2)
        J[:,2] = tf_utils.adjoint(g1.dot(g2)).dot(self.arm_calib_data.xi3)
        J[:,3] = tf_utils.adjoint(g1.dot(g2).dot(g3)).dot(self.arm_calib_data.xi4)
        J[:,4] = tf_utils.adjoint(g1.dot(g2).dot(g3).dot(g4)).dot(self.arm_calib_data.xi5)
        J[:,5] = tf_utils.adjoint(g1.dot(g2).dot(g3).dot(g4).dot(g5)).dot(self.arm_calib_data.xi6)
        return J
    
    def distance(self, query):
        distT = np.linalg.norm(query[:3, 3] - self.target[:3, 3])
        queryR  = Quaternion(matrix=query)
        targetR = Quaternion(matrix=self.target)
        distR = Quaternion.absolute_distance(queryR,targetR)
        return self.wT*distT + self.wR*distR
            
    def opt_func(self,X):
        d = self.distance(
            get_tip_pose(X, self.arm_calib_data)
        )
        return d

    def get_goal_jp(self, init_jp):
        # start = time.time()
        # x = least_squares(
        #     self.opt_func,
        #     init_jp,
        #     bounds=self.constraints,
        #     # max_nfev=10000,
        #     # xtol=None,
        #     # ftol=None,
        #     # loss="linear"
        # ).x

        x = minimize(
            self.opt_func,
            init_jp,
            # method='SLSQP',
            tol=1e-10,
            bounds=self.constraints
        ).x
        # print('Elapsed: %s' % (time.time() - start))
        return x
    
    def opt_func_jaw(self,X):
        # return distance(
        #     get_tip_pose_jaw(X[:6], self.arm_calib_data, self.g_psmtip_psmjaw),
        #     self.target,
        #     X[6],
        #     X[7]
        # )
        return self.distance(
            get_tip_pose_jaw(X, self.arm_calib_data),
        )
    
    def get_goal_jp_jaw(self, init_jp):
        # x = least_squares(
        #     self.opt_func_jaw,
        #     x0,
        #     bounds=self.constraints,
        #     # max_nfev=10000,
        #     # xtol=None,
        #     # ftol=None,
        #     # loss="linear"
        # ).x
        x = minimize(
            self.opt_func_jaw, 
            init_jp, 
            # method='SLSQP',
            tol=1e-10,
            bounds=self.constraints
        ).x
        return x
    