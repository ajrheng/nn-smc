import numpy as np
import math


def weighted_std(values, weights):

    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return math.sqrt(variance)


def prob_zero(omega, phi, t):
    return np.cos((omega+phi)*t/2) **2


def prob_one(omega, phi, t):
    return np.sin((omega+phi)*t/2) **2


def deriv_log_prob_zero(omega, phi, t):
    #likelihood = prob_zero(omega, phi, t)
    #return np.multiply(1/likelihood, -t/2 * np.sin((omega+phi)*t))
    return -np.tan((omega+phi)*t/2) * t


def deriv_log_prob_one(omega, phi, t):
    #likelihood = prob_one(omega, phi, t)
    #return np.multiply(1/likelihood, t/2 * np.sin((omega+phi)*t))
    return 1/np.tan((omega+phi)*t/2) * t


def second_deriv_log_prob_zero(omega, phi, t):
    return - (t**2)/2 * 1/ (np.cos((omega+phi)*t/2)**2)


def second_deriv_log_prob_one(omega, phi, t):
    return - (t**2)/2 * 1/ (np.sin((omega+phi)*t/2)**2)
