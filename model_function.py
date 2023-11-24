import numpy as np
import matplotlib.pyplot as plt


def double_exponential_function( t: np.ndarray,parameter_array: np.ndarray )->np.ndarray:
    """Evaluate double exponential function A1*e^(t*t1)+A2*e^(t*t2)+j0 over the array of t with given parameters.
    Returns an array with the same shape as t. Function value is calculated element-wise."""
    assert parameter_array.shape == (5,) , f"shape of parameter array is {parameter_array.shape}, expected {(5,)}"
    A1,t1,A2,t2,j0 = parameter_array

    return A1*np.exp(-(t)*t1)+A2*np.exp(-(t)*t2)+j0

