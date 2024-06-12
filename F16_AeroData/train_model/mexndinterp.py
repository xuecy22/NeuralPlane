import torch
import time
import numpy as np
import cProfile

profile = cProfile.Profile()
device = "cuda:0"


def getHyperCube(X, V, ndinfo):
    n = len(V[0])
    indexMatrix = torch.zeros((len(ndinfo), n, 2), dtype=int, device=device)
    for i, indexMax in enumerate(ndinfo):
        xmax = X[i][indexMax - 1]
        xmin = X[i][0]
        matX = X[i].repeat(n, 1)
        x = V[i].reshape(-1, 1)
        # import pdb; pdb.set_trace()
        if torch.any((x - xmin) < 0) or torch.any((x - xmax) > 0):
            print('Point lies out data grid (in getHyperCube)')
        else:
            deltaX = matX - x
            mask1 = deltaX == 0
            index1 = torch.nonzero(mask1)
            if torch.any(mask1):
                indexMatrix[i, index1[:, 0], 0] = index1[:, 1]
                indexMatrix[i, index1[:, 0], 1] = index1[:, 1]
            mask2 = deltaX > 0
            mask2[index1[:, 0], :] = True
            mask3 = torch.logical_xor(mask2[:, 1:], mask2[:, :-1])
            index2 = torch.nonzero(mask3)
            if torch.any(mask3):
                indexMatrix[i, index2[:, 0], 0] = index2[:, 1]
                indexMatrix[i, index2[:, 0], 1] = index2[:, 1] + 1
    return indexMatrix


def getLinIndex(indexVector, ndinfo):
    n = indexVector.shape[1]
    linIndex = torch.zeros(n, dtype=int, device=device)
    nDimension = len(ndinfo)
    for i in range(nDimension):
        P = 1
        for j in range(i):
            P *= ndinfo[j]
        linIndex += P * indexVector[i, :]
    return linIndex


def linearInterpolate(T_val, V, X, ndinfo):
    n = T_val.shape[1]
    nDimension = len(ndinfo)
    nVertices = 1 << nDimension
    indexVector = torch.zeros((nDimension, n), dtype=int, device=device)
    oldT = T_val.clone().detach().requires_grad_(True)
    dimNum = 0
    while nDimension > 0:
        m = nDimension - 1
        nVertices = 1 << m
        newT = torch.zeros((nVertices, n), device=device)
        for i in range(nVertices):
            for j in range(m):
                mask = 1 << j
                indexVector[j, :] = (mask & i) >> j
            index1 = torch.zeros(n, dtype=int, device=device)
            index2 = torch.zeros(n, dtype=int, device=device)
            for j in range(m):
                index1 = index1 + (1 << (j + 1)) * indexVector[j, :]
                index2 = index2 + (1 << j) * indexVector[j, :]
            f1 = oldT[index1, :]
            f2 = oldT[index1 + 1, :]
            mask1 = X[dimNum, :, 0] == X[dimNum, :, 1]
            mask2 = X[dimNum, :, 0] != X[dimNum, :, 1]
            lambda_val = (V[dimNum] - X[dimNum, :, 0]) / (X[dimNum, :, 1] - X[dimNum, :, 0] + 1e-6 * mask1)
            newT[index2, :] = ((lambda_val * f2 + (1 - lambda_val) * f1) * mask2).to(torch.float32)
            newT[index2, :] = newT[index2, :] + f1 * mask1
        oldT = newT.clone().detach().requires_grad_(True)
        nDimension = m
        dimNum += 1
    result = oldT[0, :]
    return result


def interpn(X, Y, x, ndinfo):
    global profile
    profile.enable()
    n = len(x[0])
    nDimension = len(ndinfo)
    matY = Y.repeat(n, 1)
    indexVector = torch.zeros((nDimension, n), dtype=int, device=device)
    xPoint = torch.zeros((nDimension, n, 2), device=device)
    indexMatrix = getHyperCube(X, x, ndinfo)
    nVertices = 1 << nDimension
    T_val = torch.zeros((nVertices, n), device=device)
    for i in range(nDimension):
        low = indexMatrix[i, :, 0]
        high = indexMatrix[i, :, 1]
        matX = X[i].repeat(n, 1)
        xPoint[i, :, 0] = matX[np.arange(n), low]
        xPoint[i, :, 1] = matX[np.arange(n), high]
    for i in range(nVertices):
        for j in range(nDimension):
            mask = 1 << j
            val = (mask & i) >> j
            indexVector[j, :] = indexMatrix[j, :, val]
        index = getLinIndex(indexVector, ndinfo)
        T_val[i, :] = matY[np.arange(n), index]
    result = linearInterpolate(T_val, x, xPoint, ndinfo)
    profile.disable()
    return result
