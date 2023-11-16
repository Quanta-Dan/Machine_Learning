import numpy as np
import matplotlib.pyplot as plt


def double_exponential_function( t: np.ndarray, A1:float, A2: float, t1: float, t2: float,toff1: float, toff2: float, j0: float )->np.ndarray:
    """Evaluate double exponential function A1*e^(t/t1)+A2*e^(t/t2)+j0 over the array of t with given parameters.
    Returns an array with the same shape as t. Function value is calculated element-wise."""
    return A1*np.exp(-(t+toff1)/t1)+A2*np.exp(-(t+toff2)/t2)+j0

