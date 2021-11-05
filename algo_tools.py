import numpy as np
from cvxopt import matrix, solvers
from scipy.stats import bernoulli


def transition(states, action):
    """Transition function when considering only the positive states.

    Args:
      states (int[]): Array of size k, where at index i the array keeps
      track of the state of arm i
      action (int): The index of the action that has been played
    Returns:
      states (int[]): States of each action."""
    for arm in range(len(states)):
        if arm == action:
            states[arm] = 1
        else:
            states[arm] += 1
    return states


def gen_c_x(k, d, t, M_x):
    """Function that generates c related to matrix F in the objective
    function of the LP program.

    Args:
      k: Number of actions
      d: Size of the block
      t: Current time step
      M_x: matrix M related to F
    Returns:
      c_x related to F."""
    c_x = M_x[:, t:].ravel()
    return c_x


def gen_c_y(M, d, t):
    """Function that generates c related to matrix Y in the objective
    function of the LP program.

    Args:
      M: matrix M related to Y
      d: Size of the block
      t: Current time step
    Retunrs:
      c_y related to Y."""
    c_y = np.repeat(M[:, :, np.newaxis], d - t, axis=2).ravel()
    return c_y


def gen_A_x(k, d, t):
    """Function that generates the A matrix (for the Ax=b equality)
    related to F in the objective function of the LP program.

    Args:
      k: Number of actions
      d: Size of the block
      t: Current time step
    Returns:
      A_x related to F."""
    A_x = np.repeat(np.eye(d - t)[:, np.newaxis, :],
                    k, axis=1).reshape(d - t, -1)
    return A_x


def gen_A_y(k, d, t):
    """Function that generates the A matrix (for the Ax=b equality)
    related to Y in the objective function of the LP program.

    Args:
      k: Number of actions
      d: Size of the block
      t: Current time step
    Returns:
      A_y related to Y."""
    A_y = np.repeat(np.eye(d - t)[:, np.newaxis, :],
                    k * d, axis=1).reshape(d - t, -1)
    # Equivalent to the following two-steps operation:
    # A = np.repeat(np.eye(d - t)[:, np.newaxis, :], k, axis=1)
    # A = np.repeat(A[:, :, np.newaxis, :, d, axis=2])
    # A = A.reshape(d - t, -1)
    return A_y


def gen_b(d, t):
    """Function that generates the b vector for the Ax=b equality in the
    LP program.

    Args:
      d: Size of the block
      t: Current time step
    Returns:
      The b vector."""
    b = np.ones(d - t)
    return b


def gen_G1_x(k, d, t):
    """Function that generates the G1 part of the G matrix related to F.

    Args:
      k: Number of actions
      d: Size of the block
      t: Current time step
    Returns:
      G1_x, related to F."""
    G1_x = np.repeat(np.eye(k)[:, :, np.newaxis],
                     d - t, axis=2).reshape(k, -1)
    return G1_x


def gen_G1_y(k, d, t):
    """Function that generates the G1 part of the G matrix related to Y.

    Args:
      k: Number of actions
      d: Size of the block
      t: Current time step
    Returns:
      G1_y, related to Y."""
    G1_y = np.zeros((k, k * d * (d - t)))
    return G1_y


def gen_G2_x(k, d, t):
    """Function that generates the G2 part of the G matrix related to F.

    Args:
      k: Number of actions
      d: Size of the block
      t: Current time step
    Returns:
      G2_x, related to F."""
    A = np.eye(k)
    B = np.tril(np.ones((d - t)), -1)
    G2_x = A[:, np.newaxis, :, np.newaxis] * B[np.newaxis, :, np.newaxis, :]
    G2_x = G2_x.reshape(k * (d - t), -1)
    return - G2_x


def gen_G2_y(k, d, t):
    """Function that generates the G2 part of the G matrix related to Y.

    Args:
      k: Number of actions
      d: Size of the block
      t: Current time step
    Returns:
      G2_x, related to Y."""
    A = np.eye(k)
    B = np.eye(d - t)
    C = A[:, np.newaxis, :, np.newaxis] * B[np.newaxis, :, np.newaxis, :]
    G2_y = np.repeat(C[:, :, :, np.newaxis, :], d, axis=3)
    G2_y = G2_y.reshape(k * (d - t), -1)
    return G2_y


def gen_G3_x(k, d, t):
    """Function that generates the G3 part of the G matrix related to F.

    Args:
      k: Number of actions
      d: Size of the block
      t: Current time step
    Returns:
      G3_x, related to F."""
    A = np.eye(k)
    B = np.zeros((d, d, d))
    for p in range(1, d + 1):
        e = np.zeros((d))
        e[p - 1] = 1
        C = np.tril(np.ones((d, d)), -p) - np.tril(np.ones((d, d)), -(p + 1))
        B += e[:, np.newaxis, np.newaxis] * C[np.newaxis, :, :]
    G3_x = A[:, np.newaxis, np.newaxis, :, np.newaxis] * B[np.newaxis, :, :,
                                                           np.newaxis, :]
    G3_x = G3_x.reshape(k * d * (d - t), -1)
    return - G3_x


def gen_G3_y1(k, d, t):
    """Function that generates a specific part of the G3 matrix in G
    related to Y.

    Args:
      k: Number of actions
      d: Size of the block
      t: Current time step
    Returns:
      G3_y1, related to Y."""
    A = np.eye(k)
    B = np.eye(d)
    C = np.eye(d)
    G3_y1 = A[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis] * \
        B[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis] * \
        C[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :]
    G3_y1 = G3_y1.reshape(k * d * (d - t), -1)
    return G3_y1


def gen_G3_y2(k, d, t):
    """Function that generates a specific part of the G3 matrix in G
    related to Y.

    Args:
      k: Number of actions
      d: Size of the block
      t: Current time step
    Returns:
      G3_y2, related to Y."""
    A = np.eye(k)
    B = np.zeros((d, d, d))
    for p in range(1, d + 1):
        e = np.zeros((d))
        e[p - 1] = 1
        C = np.tril(np.ones((d, d)), -p) - np.tril(np.ones((d, d)), -(p + 1))
        B += e[:, np.newaxis, np.newaxis] * C[np.newaxis, :, :]
    G3_y2 = A[:, np.newaxis, np.newaxis, :, np.newaxis] * B[np.newaxis, :,
                                                            :, np.newaxis, :]
    G3_y2 = np.repeat(G3_y2[:, :, :, :, np.newaxis, :], d, axis=4)
    G3_y2 = G3_y2.reshape(k * d * (d - t), -1)
    return G3_y2


def gen_G3_y3(k, d, t):
    """Function that generates a specific part of the G3 matrix in G
    related to Y.

    Args:
      k: Number of actions
      d: Size of the block
      t: Current time step
    Returns:
      G3_y3, related to Y."""
    A = np.eye(k)
    B = np.zeros((d, d, d, d))
    for p in range(1, d + 1):  # be careful here
        C = np.tril(np.ones((d, d)), -p) - np.tril(np.ones((d, d)), -(p + 1))
        B += C[:, np.newaxis, :, np.newaxis] * C[np.newaxis, :, np.newaxis, :]
    G3_y3 = A[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis] *\
        B[np.newaxis, :, :, np.newaxis, :, :]
    G3_y3 = G3_y3.reshape(k * d * (d - t), -1)
    return G3_y3


def gen_G3_y(k, d, t):
    """Function that generates the entire G3 matrix in G
    related to Y.

    Args:
      k: Number of actions
      d: Size of the block
      t: Current time step
    Returns:
      G3_y, related to Y."""
    G3_y1 = gen_G3_y1(k, d, t)
    G3_y2 = gen_G3_y2(k, d, t)
    G3_y3 = gen_G3_y3(k, d, t)
    G3_y = G3_y1 - G3_y2 + G3_y3
    return G3_y


def gen_G_full(k, d, t):
    """Function that generates the entire G matrix.

    Args:
      k: Number of actions
      d: Size of the block
      t: Current time step
    Returns:
      The full G matrix."""
    G1_x = gen_G1_x(k, d, t)
    G1_y = gen_G1_y(k, d, t)
    G1 = np.hstack((G1_x, G1_y))
    G2_x = gen_G2_x(k, d, t)
    G2_y = gen_G2_y(k, d, t)
    G2 = np.hstack((G2_x, G2_y))
    G3_x = gen_G3_x(k, d, t)
    G3_y = gen_G3_y(k, d, t)
    G3 = np.hstack((G3_x, G3_y))
    new_dim = k * (d - t) * (d + 1)
    G_full = np.vstack((G1, G2, G3, -np.eye(new_dim),
                        np.eye(new_dim)))
    return G_full


def gen_h_full(k, d, t):
    """Function that generates the entire h vector.

    Args:
      k: Number of actions
      d: Size of the block
      t: Current time step
    Returns:
      The full h vector."""
    h1 = np.ones(k)
    h2 = np.zeros((k * d))
    h3 = np.zeros((k * d * d))
    new_dim = k * d * (d + 1)
    h4 = np.zeros(new_dim)
    h5 = np.ones(new_dim)
    h_full = np.hstack((h1.ravel(), h2.ravel(), h3.ravel(), h4.ravel(),
                       h5.ravel()))
    return h_full


def get_Mx_from_initial_states(M, states):
    """Function that generates the M matrix, which stores the reward for each
    action-state, for the first pulls in the sequence looking at the initial
    states. This function is particularly useful for CombUCB1 and to retrieve
    the actual reward of an action when its state is greater than the size
    of the block, d.

    Args:
      M: The original M matrix
      states: states of the actions
    Returns:
      The M_x matrix."""
    k, d = M.shape
    M_x = np.zeros((k, d))
    for i in range(len(states)):
        state = states[i]
        M_x[i, :] = np.append(M[i, state - 1:],
                              np.ones((d - M[i, state - 1:].shape[0])))
    return M_x


def seq_from_XYtens(xtens, ytens, d):
    """Given the tensor representation of F and Y, computes the sequence
    of actions.

    Args:
      xtens: Tensor representation of F
      ytens: Tensor representation of Y
      d: Size of the block
    Returns:
      seq: Sequence of action represented by the tensor representation."""
    # if problems, check if xtens and ytens contain integers or not
    k, t = xtens.shape
    seq = np.zeros(t)
    for i in range(t):
        slice_i = np.append(xtens[:, i], ytens[:, :, i])
        imax = np.argmax(slice_i)
        if imax < k:
            seq[i] = imax
        else:
            seq[i] = (imax - k) // d
    return seq


def XYtens_from_seq(seq, k, d, init_states):
    """Given the sequence of actions, computes the tensor representation
       of F and Y.

    Args:
      seq: Sequence of actions
      k: Number of actions
      d: Size of the block
      init_states: Array storing the initial states for each action
    Returns:
      X: Tensor representation of F
      Y: Tensor representation of Y."""
    t = len(seq)
    X = np.zeros((k, t))
    Y = np.zeros((k, d, t))
    states = []
    states = init_states[:]

    for i in range(t):

        arm = seq[i]
        if X[arm, :].sum() < 0.5:
            X[arm, i] = 1
            states = transition(states, arm)
        else:
            delay = states[arm]
            Y[arm, delay - 1, i] = 1
            states = transition(states, arm)

    return X, Y


def gen_G_t_top(k, d, t, G_full_0):
    """Function that generates part of the G matrix.

    Args:
      k: Number of arms
      d: Size of the block
      t: Current time steps
      G_full_0: Upper part of the G matrix
    Returns:
      Multiple parts of the remaining G matrix."""
    G_full_0_top = G_full_0[:k + k * d + k * d * d, :]

    # select all G1
    G1 = G_full_0_top[:k, :]

    G1x = G1[:, :k * d].reshape(k, k, d)
    G1x_past = G1x[:, :, :t]
    G1x_fut = G1x[:, :, t:]

    G1y = G1[:, k * d:].reshape(k, k, d, d)
    G1y_past = G1y[:, :, :, :t]
    G1y_fut = G1y[:, :, :, t:]

    # select all G2
    G2 = G_full_0_top[k:k + k * d, :]

    G2x = G2[:, :k * d].reshape(k, d, k, d)
    G2x_past = G2x[:, t:, :, :t]
    G2x_fut = G2x[:, t:, :, t:]

    G2y = G2[:, k * d:].reshape(k, d, k, d, d)
    G2y_past = G2y[:, t:, :, :, :t]
    G2y_fut = G2y[:, t:, :, :, t:]

    # select all G3
    G3 = G_full_0_top[k + k * d:, :]

    G3x = G3[:, :k * d].reshape(k, d, d, k, d)
    G3x_past = G3x[:, :, t:, :, :t]
    G3x_fut = G3x[:, :, t:, :, t:]

    G3y = G3[:, k * d:].reshape(k, d, d, k, d, d)
    G3y_past = G3y[:, :, t:, :, :, :t]
    G3y_fut = G3y[:, :, t:, :, :, t:]

    # create Gx future
    Gx_fut = np.vstack((G1x_fut.reshape(k, k * (d - t)),
                        G2x_fut.reshape(k * (d - t), k * (d - t)),
                        G3x_fut.reshape(k * d * (d - t), k * (d - t))))

    # create Gy future
    Gy_fut = np.vstack((G1y_fut.reshape(k, k * d * (d - t)),
                        G2y_fut.reshape(k * (d - t), k * d * (d - t)),
                        G3y_fut.reshape(k * d * (d - t), k * d * (d - t))))

    # recombine to create the full G future
    G_fut = np.hstack((Gx_fut, Gy_fut))

    return G_fut, G1x_past, G1y_past, G2x_past, G2y_past, G3x_past, G3y_past


def gen_G_t(k, d, t, G_full_0):
    """Compute the G matrix for the current time step t.

    Args:
      k: Number of actions
      d: Size of the block
      t: Current time step
      G_full_0: full matrix of the G0 part of G
    Returns:
      G_full: the full G matrix."""
    G_fut, _, _, _, _, _, _ = gen_G_t_top(k, d, t, G_full_0)
    new_dim = k * (d - t) * (d + 1)
    G_full = np.vstack((G_fut, -np.eye(new_dim), np.eye(new_dim)))

    return G_full


def gen_h_full_0(k, d, t):
    """Compute the h1, h2, h3 parts of the h vector.

    Args:
      k: Number of actions
      d: Size of the block
      t: Current time step
    Returns:
      h1, h2, h3 parts of the vector h."""
    h1 = np.ones(k)
    h2 = np.zeros((k * (d - t)))
    h3 = np.zeros((k * d * (d - t)))
    return h1, h2, h3


def gen_h_full_t(k, d, t, G_full_0=None, X=None, Y=None):
    """Compute the full h vector.

    Args:
      k: Number of actions
      d: Size of the block
      t: Current time step
      G_full_0: full matrix of the G0 part of G
      X: tensor for first pulls
      Y: tensor for non-first pulls
    Returns:
      h_full: the full h vector."""
    h1, h2, h3 = gen_h_full_0(k, d, t)

    if t > 0.5:

        G_fut, G1x_past, G1y_past, G2x_past, G2y_past, G3x_past, \
            G3y_past = gen_G_t_top(k, d, t, G_full_0)

        G1x_past = G1x_past.reshape(k, -1)
        G1y_past = G1y_past.reshape(k, -1)
        G2x_past = G2x_past.reshape((k * (d - t)), -1)
        G2y_past = G2y_past.reshape((k * (d - t)), -1)
        G3x_past = G3x_past.reshape((k * d * (d - t)), -1)
        G3y_past = G3y_past.reshape((k * d * (d - t)), -1)

        h1 -= (np.dot(G1x_past, X) + np.dot(G1y_past, Y))
        h2 -= (np.dot(G2x_past, X) + np.dot(G2y_past, Y))
        h3 -= (np.dot(G3x_past, X) + np.dot(G3y_past, Y))

    new_dim = k * (d - t) * (d + 1)
    h4 = np.zeros(new_dim)
    h5 = np.ones(new_dim)
    h_full = np.hstack((h1.ravel(), h2.ravel(), h3.ravel(), h4.ravel(),
                        h5.ravel()))
    return h_full


def reward_seq_solver(seq, M_x, init_states, M):
    t = len(seq)
    k = M.shape[0]
    rwd = 0
    first_pulls = np.zeros(k, dtype=int)
    states = []
    states = init_states[:]

    for i in range(t):
        arm = seq[i]
        if first_pulls[arm] == 0:
            rwd += M_x[arm, i]
            transition(states, seq[i])
            first_pulls[arm] = 1
        else:
            rwd += M[arm, states[arm] - 1]
            transition(states, seq[i])
    return rwd


def oracle(k, d, initial_states, M, M_x):
    """Solver that computes the solution to the LP program.

    Args:
      k: Number of actions
      d: size of the block
      initial_states: vector which stores the initial states of the actions
      M: reward matrix for non-first pulls
      M_x: reward matrix for first pulls
    Returns:
      seq: sequence of actions which is the solution returned by the solver."""
    seq = []
    t = len(seq)
    init_states = list(initial_states)
    c_x = M_x.ravel()
    c_y = gen_c_y(M, d, t)
    c = np.hstack((c_x, c_y))

    # gen of A
    A_x = gen_A_x(k, d, t)
    A_y = gen_A_y(k, d, t)
    A = np.hstack((A_x, A_y))
    # gen of b
    b = gen_b(d, t)

    # gen of G
    G_full_0 = gen_G_full(k, d, t)
    # gen of h
    h_full_0 = gen_h_full(k, d, t)

    dims = {'l': G_full_0.shape[0], 'q': [], 's': []}
    solvers.options['show_progress'] = False
    sol = solvers.conelp(matrix(-c), matrix(G_full_0), matrix(h_full_0),
                         dims, matrix(A), matrix(b))
    x = np.array(sol['x'])

    for time in range(d - 1):
        rewards = np.zeros(k)
        for arm in range(k):
            seq.append(arm)
            t = len(seq)
            X_t, Y_t = XYtens_from_seq(seq, k, d, init_states)

            c_x = M_x[:, t:].ravel()
            c_y = gen_c_y(M, d, t)
            c = np.hstack((c_x, c_y))
            A_x = gen_A_x(k, d, t)
            A_y = gen_A_y(k, d, t)
            A = np.hstack((A_x, A_y))
            b = gen_b(d, t)

            G_full = gen_G_t(k, d, t, G_full_0)
            h_full = gen_h_full_t(k, d, t, G_full_0, X_t.ravel(), Y_t.ravel())

            # We run the solver calling CVXOPT
            dims = {'l': G_full.shape[0], 'q': [], 's': []}
            solvers.options['show_progress'] = False
            sol = solvers.conelp(matrix(-c), matrix(G_full), matrix(h_full),
                                 dims, matrix(A), matrix(b))
            x = sol['x']
            x = np.array(x)

            rewards[arm] = (reward_seq_solver(seq, M_x, init_states, M)
                            - sol['primal objective'])
            seq.pop(t - 1)
        best_arm = np.argmax(rewards)
        seq.append(best_arm)

    rewards = np.zeros(k)
    rewards = [reward_seq_solver(np.append(seq, i), M_x, init_states, M)
               for i in range(k)]
    seq.append(np.argmax(rewards))
    return seq


def update_UCBs(M, k, d, mean_rwds, states, times_played, t):
    """Function that computes and updates the UCB indexes of the arms.

    Args:
      M: Matrix storing the UCB indexes of the arms
      k: Number of actions
      d: Size of the block
      mean_rwds: Matrix containing the verage reward for each arm and state
      states: State of the actions
      times_played: Number of times each action has been played
      t: Current time step in the horizon
      explor: Flag variable that indicates if the exploration bonus in the
        UCB formula is considered or not."""
    curr_states = list(states)
    explor_bonus = 0.0
    for i in range(k):
        arm = i
        tau = curr_states[arm]
        if tau > d:
            tau = d
        if times_played[arm, tau - 1] > 0.5:
            explor_bonus = np.sqrt(
                (1.5 * np.log(t)) / times_played[arm, tau - 1])
            M[arm, tau - 1] = mean_rwds[arm, tau - 1] + explor_bonus
    return M


def reward_seq_XY_firstpulls(seq, init_states, M_rwd):
    """Function that computes the reward of the sequence when distinguishing
    between first pulls and non-first pulls.

    Args:
      seq: Sequence of actions
      M_x: Reward matrix for the first pulls
      init_states: Initial states of the actions
      M_rwd: Matrix containing the expected mean reward for each arm and state
    Returns:
      single_rwds: Array storing the individual rewards for each arm played
        in the block, where the element at index i corresponds to the action
        played at time step i in the block
      first_pulls: Array whose elements are 0 or 1 if the action is
      respectively a non-first pull or a first pull.
    """
    t = len(seq)
    k, d = M_rwd.shape
    first_pulls_check = np.zeros(k, dtype=int)
    first_pulls = np.zeros(t, dtype=int)
    single_rwds = np.zeros((t))
    states = list(init_states)

    for i in range(t):
        arm = seq[i]
        state = states[arm]
        if states[arm] >= d:
            state = d - 1
        if first_pulls_check[arm] == 0:
            transition(states, seq[i])
            first_pulls_check[arm] = 1
            first_pulls[i] = 1
            single_rwds[i] = bernoulli.rvs(size=1, p=M_rwd[arm, state - 1])
        else:
            single_rwds[i] = bernoulli.rvs(size=1, p=M_rwd[arm, state - 1])
            transition(states, seq[i])

    return single_rwds, first_pulls


def reward_seq_XY_CombUCB1(seq, curr_states, M_rwd):
    """Compute the reward of the sequence played by CombUCB1.

    Args:
      seq: Sequence of actions
      M_x: Matrix of rewards related to the first pulls, F.
      curr_states: Array storing the current states for each action
      M: Matrix of rewards related to non-first pulls of the action
    Returns:
      single_rwds: Array storing the individual rewards for each arm played
        in the block, where the element at index i corresponds to the action
        played at time step i in the block
      first_pulls: Array whose elements are 0 or 1 if the action is
        respectively a non-first pull or a first pull."""
    d = len(seq)
    max_state = M_rwd.shape[1] - 1
    first_pulls = np.zeros(d, dtype=int)
    single_rwds = np.zeros((d))
    init_states = list(curr_states)

    for i in range(d):
        arm = seq[i]
        if init_states[arm] > max_state:
            init_states[arm] = max_state
        single_rwds[i] = bernoulli.rvs(size=1,
                                       p=M_rwd[arm, init_states[arm] - 1])
        init_states = transition(init_states, arm)
    return single_rwds, first_pulls


def get_Mx_rwd(M, states, d):
    """Function that computes the M_x matrix from the current states.
    This function is necessary in order to be able to compute the reward matrix
    for the first pulls when considering the actual states.
    Args:
      M: Reward matrix
      states: States of actions
      d: Size of the block
    Returns:
      M_x: Matrix for first pulls."""
    k = M.shape[0]
    M_x = np.zeros((k, d))
    tau_d = np.copy(M[:, d - 1])
    M_rest = np.repeat(tau_d.transpose(), d).reshape((k, d))
    for i in range(len(states)):
        state = states[i]
        M_x[i, :] = np.append(M[i, state - 1:],
                              M_rest[i, 0: state - 1])
    return M_x


def transition_neg(states, action):
    """Transition function when considering both positive and negative states.

    Args:
      states (int[]): Array of size k, where at index i the array keeps
      track of the state of arm i
      action (int): The index of the action that has been played
    Returns:
      states (int[]): States of each action."""
    for arm in range(len(states)):
        if arm == action:
            if states[arm] < 0:
                states[arm] -= 1
            else:
                states[arm] = -1
        else:
            states[arm] += 1
            if states[arm] == 0:
                states[arm] = 1
    return states
