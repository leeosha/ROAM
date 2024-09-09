#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
sys.path.append('/home/admin')

import numpy as np
import numba
from numba import jit,int32,float64,void,int64
import random

#********************************************************************#
##以下为自定义的计算函数，使用numba jit进行了运行加速
#********************************************************************#
@jit(float64[:](float64[:,:], float64[:], float64[:], float64[:], float64[:], float64[:], float64),nopython=True)
def CalculateOpt(roam_param_array, scores, ctrs, cvrs, cpcs, prices, beta):
    alpha = roam_param_array[:, 1]
    eta = roam_param_array[:, 11]
    zeta = roam_param_array[:, 12]
    # pv 预估roi和真实累计roi的比值系数
    roi_coeff = roam_param_array[:, 14]
    # lower bound
    roi_lb = roam_param_array[:,13]*roi_coeff
    # upper bound
    roi_ub = roam_param_array[:,15]*roi_coeff
    theta = roam_param_array[:, 3]
    priority = roam_param_array[:, 4]
    cij = ctrs*cpcs
    gij = ctrs*cvrs*prices
    # xij_array = theta *(1.0+(scores - ctrs*cpcs*alpha - beta)/priority)
    xij_array = theta *(1.0+(scores - ctrs*cpcs*alpha - eta*(roi_lb*cij - gij) - zeta*(gij - roi_ub*cij) - beta)/priority)
    # xij_array = theta *(1.0+(scores - ctrs*cpcs*alpha - eta*(roi_lb*cij - gij) - beta)/priority)
    # if random.random()<0.001:
    #     print("xij:", xij_array,", theta:",theta,",scores:",scores,",alpha:",alpha,",eta:",eta,",zeta:",zeta, ",ctrs*cpcs*alpha:",ctrs*cpcs*alpha,",roi_related:",eta*(roi_lb*cij - gij) + zeta*(gij - roi_ub*cij),",beta:",beta)
    return np.maximum(0.0,xij_array)

def solve_max(coef, y):
    solutions = []
    sum_cons = 0.0
    sum_coef = 0.0
    for i in range(len(coef)):
        sum_cons += coef[i][1]
        sum_coef += coef[i][2]
    sorted_coef = sorted(coef, key=lambda t:t[0])
    res = (sum_cons - y) / sum_coef
    if res <= sorted_coef[0][0]:
        solutions.append(res)
    for i in range(1, len(sorted_coef)):
        sum_cons -= sorted_coef[i-1][1]
        sum_coef -= sorted_coef[i-1][2]
        res = (sum_cons - y) / sum_coef
        if res <= sorted_coef[i][0]:
            solutions.append(res)
            break
    return solutions

def CalculateBeta(roam_param_array , scores, ctrs, cvrs, cpcs, prices):
    coef = []
    for j in range(len(scores)):
        score = scores[j]
        ctr = ctrs[j]
        cvr = cvrs[j]
        cpc = cpcs[j]
        price = prices[j]
        theta = roam_param_array[j][3]
        alpha = roam_param_array[j][1]
        priority = roam_param_array[j][4]
        eta = roam_param_array[j][11]
        zeta = roam_param_array[j][12]
        # pv 预估roi和真实累计roi的比值系数
        roi_coeff = roam_param_array[j][14]
        # lower bound
        roi_lb = roam_param_array[j][13] * roi_coeff
        # upper bound
        roi_ub = roam_param_array[j][15] * roi_coeff
        cij = ctr * cpc
        gij = ctr * cvr * price
        # a = theta * (1.0 + (score- ctr*cpc*alpha)/ priority)
        # a = theta * (1.0 + (score- ctr*cpc*alpha- eta*(roi_lb*cij - gij) - zeta*(gij - roi_ub*cij))/ priority)
        a = theta * (1.0 + (score- ctr*cpc*alpha- eta*(roi_lb*cij - gij) )/ priority)
        b = theta / priority
        coef.append((a / b, a, b))
    result = solve_max(coef, 1.0)
    if len(result) == 0 or result[0] < 0.0:
        return 0.0
    else:
        return result[0]

@jit(float64(float64[:,:], float64[:], float64),nopython=True)
def CalculateL2NormLoss(roam_param_array,xij_array,si):
    # 计算训练过程中的分配l2normloss值
    priority = roam_param_array[:, 4]
    theta = roam_param_array[:, 3]
    l2normloss=np.sum(0.5 * si * priority * np.square(xij_array - theta) / theta)
    return l2normloss

@jit(void(float64[:,:], float64[:], float64[:],float64[:], float64[:], float64[:], float64[:], int64[:], float64[:], float64, float64[:], float64[:], float64[:], float64[:], float64[:], float64), nopython=True)
def update_grad(roam_param_array, cost, gmv, alpha_grads , eta_grads, zeta_grads, monitor_param, indexes, xijs, si, scores, ctrs, cvrs, cpcs, prices, l2normloss):
    cost[indexes] += xijs * si * ctrs * cpcs
    gmv[indexes] += xijs * si * ctrs * cvrs * prices
    alpha_grads[indexes] += si * ctrs *ctrs * cpcs * cpcs
    # pv 预估roi和真实累计roi的比值系数
    roi_coeff = roam_param_array[:, 14]
    # lower bound
    roi_lb = roam_param_array[:, 13] * roi_coeff
    # upper bound
    roi_ub = roam_param_array[:, 15] * roi_coeff
    cij = xijs * si * ctrs * cpcs
    gij = xijs * si * ctrs * cvrs * prices

    # eta_grads[indexes] += roi_lb * cij - gij
    eta_grads[indexes] += si * ctrs * cpcs * (roi_lb * cij - gij)
    # zeta_grads[indexes] += gij - roi_ub * cij
    zeta_grads[indexes] += si * ctrs * cpcs * (gij - roi_ub * cij)
    monitor_param[0] += np.sum(xijs * si * ctrs * cpcs)
    monitor_param[1] += np.sum(xijs * scores * si)
    monitor_param[2] += l2normloss


@jit(float64[:](float64[:,:], float64, int64, float64[:], float64[:], float64[:], float64[:], float64[:]),nopython=True)
def CalculateSigma(roam_param_array, beta, current_batch, scores, ctrs, cvrs, cpcs, prices):
    # 公式 xij=roam_param['theta']*(1.0+(1.0 + lambda_ * roam_param['pctr'] - alpha - beta)/priority)
    xij_remain = 1.0
    batch_id = roam_param_array[:,8]
    theta = roam_param_array[:,3]
    penality = roam_param_array[:,10]
    alpha = roam_param_array[:,1]
    sigma = roam_param_array[:, 2]
    priority = roam_param_array[:,4]
    eta = roam_param_array[:, 11]
    zeta = roam_param_array[:, 12]
    # pv 预估roi和真实累计roi的比值系数
    roi_coeff = roam_param_array[:, 14]
    # lower bound
    roi_lb = roam_param_array[:, 13] * roi_coeff
    # upper bound
    roi_ub = roam_param_array[:, 15] * roi_coeff
    cij = ctrs * cpcs
    gij = ctrs * cvrs * prices

    # for batch_id < current_batch:
    xij_array_high_priority = np.where(batch_id < current_batch, theta *(1+( scores - ctrs*cpcs*sigma - eta*(roi_lb*cij - gij) - zeta*(gij - roi_ub*cij) - beta)/priority), 0)
    xij_array_high_priority = np.maximum(0.0, xij_array_high_priority)
    if xij_array_high_priority.sum() >= 1.0:
        return xij_array_high_priority / xij_array_high_priority.sum() * 1
    else :
        xij_remain = max(0.0, xij_remain - xij_array_high_priority.sum())
        # for batch_id >= current_batch:
        xij_array_low_priority = np.where(batch_id >= current_batch, theta *(1+( scores - ctrs*cpcs*sigma - eta*(roi_lb*cij - gij) - zeta*(gij - roi_ub*cij) - beta)/priority), 0)
        xij_array_low_priority = np.maximum(0.0, xij_array_low_priority)
        if xij_array_low_priority.sum() > xij_remain:
            xij_array_low_priority = xij_array_low_priority / xij_array_low_priority.sum() * xij_remain
        return np.maximum(0.0,xij_array_high_priority + xij_array_low_priority)
