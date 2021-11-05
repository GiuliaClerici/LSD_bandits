import numpy as np


def gen_M_trickCSBOG(k, d, pos):
    """Compute the M matrix of a particular instance where ISI-CombUCB1
       performs better than both CSB and Oracle Greedy.

    Args:
      k: Number of arms
      d: Size of the block
      pos: State where the reward function of arm 1 has a spike
    Returns:
      M: the M matrix storing the mu_i(j) of the arms."""
    M = np.zeros((k, d))
    for i in range(k):
        for j in range(d):
            if i == 1:
                if j == pos - 1:
                    M[i, j] = 0.95
                else:
                    M[i, j] = 0.0
            elif i == 2:
                if j >= (pos * 3) - 1:
                    M[i, j] = 0.96
                elif j == (pos * 2) - 1:
                    M[i, j] = 0.16
                else:
                    M[i, j] = 0.14
            else:
                M[i, j] = 0.15
    return M


def gen_M_trickOG(k, d, pos):
    """Compute the M matrix of a particular instance where ISI-CombUCB1
       performs better than Oracle Greedy.

    Args:
      k: Number of arms
      d: Size of the block
      pos: Position of the step in the step function
    Returns:
      M: the M matrix storing the mu_i(j) of the arms."""
    M = np.zeros((k, d))
    for i in range(k):
        for j in range(d):
            if i == 0:
                M[i, j] = 0.05
            elif i == 1:
                if j == 0:
                    M[i, j] = 0.06
                elif j >= pos - 1:
                    M[i, j] = 0.95
    return M
