import numpy as np
from data_tools import gen_M_trickOG
from algo_tools import (get_Mx_rwd, oracle, reward_seq_XY_CombUCB1,
                        update_UCBs, transition, reward_seq_XY_firstpulls)
import pickle
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import matplotlib


# Initialize the parameters of the LSD bandit problem we are testing
k = 2  # Number of actions
pos = 2  # Position of the step in the step function
size_M = 11  # Specify the length of the matrix - we chose 11 since the
# length of the block for LSD is d+1 and d is 10
M_true = gen_M_trickOG(k, size_M, pos)  # Generating the reward matrix
# M_true is the matrix storing the expected average reward
# for each action and state
print("M: ", M_true)
T = 3300  # Horizon - originally 5060
# We make sure that the horizon is divisible for the size of the block
# of every algorithm we are going to test

# Testing each algorithm on 10 runs, each one with a different random seed
n_runs = 1
for seed in range(n_runs):
    np.random.seed(seed)

    # --------------------------------- Run CombUCB1 -------------------------
    print("Running Vanilla CombUCB1...")
    d = 10  # Size of the block for vanilla CombUCB1
    states = np.ones((k), dtype=int)  # Initial states of the actions
    # All arms start with initial state at 1
    s = np.zeros((k, d), dtype=int)  # Matrix where the entries indicate the
    # number of times an action mu_i(j) has been played
    rwds_mean = np.zeros((k, d))  # Average reward of each arm-state mu_i(j)
    solution_CSB = []  # Array storing all the pulls CombUCB1 plays over the
    # horizon
    tot_rwd_CSB = 0.0  # Cumulative reward of CombUCB1 over the horizon
    plot_rwds_CSB = np.zeros((T))  # Array storing the individual rewards
    # collected in each time step

    init_val = d * (np.sqrt((1.5 * np.log(k * d)) / 1))  # Initial value we use
    # to intialize the matrix containing the UCB indexes
    UCBs = np.ones((k, d)) * init_val  # Matrix storing the UCB indexes for
    # each mu_i(j)

    for t in range(T // d):
        # Get the Mx for the solver
        UCBs_x = get_Mx_rwd(UCBs[:, 0:d], states, d)
        # Run the solver of the LP program
        seq = oracle(k, d, states, UCBs, UCBs_x)
        # Compute the rewards of arms pulled
        individual_rwds, first_pulls = reward_seq_XY_CombUCB1(
            seq, states, M_true)
        # Sum the rewards obtained to the total rwd of the algorithm
        tot_rwd_CSB += individual_rwds.sum()

        # For each action we play
        for i in range(d):
            arm = seq[i]
            arm_state = states[arm]  # Select its state
            # Append the rwd to the individual rwds collected by the algorithm
            plot_rwds_CSB[(t * d) + i] = plot_rwds_CSB[(t * d) + i - 1
                                                       ] + individual_rwds[i]
            # If the state of the arm is greater than d, then we are going to
            # update the estimate for arm played with state d, mu_i(d)
            if arm_state > d:
                arm_state = d
            # For the action pulled, update the count of number of
            # plays for this action
            s[arm, arm_state - 1] += 1
            # Update the average reward of the arm
            if s[arm, arm_state - 1] > 1:
                rwds_mean[arm, arm_state - 1] = ((s[arm, arm_state - 1] - 1
                                                  ) * rwds_mean[arm,
                                                                arm_state - 1]
                                                 + individual_rwds[i]
                                                 ) / s[arm, arm_state - 1]
            else:
                rwds_mean[arm, arm_state - 1] = individual_rwds[i]
            # Update the UCB indexes
            UCBs = update_UCBs(UCBs, k, d, rwds_mean, states,
                               s, t * d + i + 1)
            # Update the states of the actions
            states = transition(states, arm)
        # Collect the sequence of pulls computed by the solver
        solution_CSB.append(seq)
    # print("Solution: ", solution_CSB)
    # print("Total reward: ", tot_rwd_CSB)

    # ------------------------------- Run ISI-CombUCB1 -----------------------
    print("Running ISI-CombUCB1...")
    d = d + 1  # LSD consider a block of size d + 1 w.r.t. CombUCB1
    states = np.ones((k), dtype=int)  # Initial states of the actions
    # All arms start with initial state at 1
    s = np.zeros((k, d), dtype=int)  # Matrix where the entries indicate the
    # number of times an action mu_i(j) has been played
    rwds_mean = np.zeros((k, d))  # Average reward of each arm-state mu_i(j)
    solution_LSD = []  # Array storing all the pulls LSD plays over the
    # horizon
    tot_rwd_LSD = 0.0  # Cumulative reward of LSD over the horizon
    plot_rwds_LSD = np.zeros((T))  # Array storing the individual rewards
    # collected in each time step

    init_val = d * (np.sqrt((1.5 * np.log(k * d)) / 1))  # Initial value we use
    # to intialize the matrix containing the UCB indexes
    UCBs = np.ones((k, d)) * init_val  # Matrix storing the UCB indexes for
    # each mu_i(j)

    for t in range(T // d):
        # Get the Mx for LSD - all zeros since we don't consider the
        # rewards of the first pulls
        UCBs_x = np.zeros((k, d))
        # Run the solver of the LP program
        seq = oracle(k, d, states, UCBs, UCBs_x)
        # Compute the rewards of arms pulled
        individual_rwds, first_pulls = reward_seq_XY_firstpulls(
            seq, states, M_true)
        # Sum the rewards obtained to the total rwd of the algorithm
        tot_rwd_LSD += individual_rwds.sum()

        for i in range(d):
            arm = seq[i]
            # Append the rwd to the individual rwds collected by the algorithm
            plot_rwds_LSD[(t * d) + i] = plot_rwds_LSD[(t * d) + i - 1
                                                       ] + individual_rwds[i]
            # If it's not a first pull, we update the UCB index
            if first_pulls[i] == 0:
                arm_state = states[arm]  # Get the state of the arm
                # For the action pulled, update the count of number of
                # plays for this action
                s[arm, arm_state - 1] += 1
                # Update the average reward of the arm
                if s[arm, arm_state - 1] > 1:
                    rwds_mean[arm, arm_state - 1] = (
                        (s[arm, arm_state - 1] - 1)
                        * rwds_mean[arm, arm_state - 1]
                        + individual_rwds[i]) / s[arm, arm_state - 1]
                else:
                    rwds_mean[arm, arm_state - 1] = individual_rwds[i]
                # Update the UCB indexes
                UCBs = update_UCBs(UCBs, k, d, rwds_mean, states,
                                   s, t * d + i + 1)
            # Update the states of the actions
            states = transition(states, arm)
        # Collect the sequence of pulls computed by the solver
        solution_LSD.append(seq)
    # print("Solution: ", solution_LSD)
    # print("Total reward: ", tot_rwd_LSD)

    # --------------------------------- Run Greedy --------------------------
    print("Running Oracle Greedy...")
    M = np.copy(M_true)  # Initial M matrix
    states = np.ones((k), dtype=int)  # Initial states of the actions
    # All arms start with initial state at 1
    s = np.zeros((k, d), dtype=int)  # Matrix where the entries indicate the
    # number of times an action mu_i(j) has been played
    rwds_mean = np.zeros((k, d))  # Average reward of each arm-state mu_i(j)
    solution_OG = []  # Array storing all the pulls Greedy plays over the
    # horizon
    tot_rwd_OG = 0.0  # Cumulative reward of Greedy over the horizon
    plot_rwds_OG = np.zeros((T))  # Array storing the individual rewards
    # collected in each time step

    for t in range(T):

        for j in range(k):
            # If there is an arm with state that exceeds the shape
            # of the M matrix, look at the last entry.
            # We assume that the entries exceeding the last state
            # codified in the M matrix are equal to the last entry
            if states[j] > M.shape[1] - 1:
                states[j] = M.shape[1] - 1
        # Look at the current \mu_i(j)
        curr_arms = np.array([M_true[i, states[i] - 1] for i in range(k)])
        # Select the max reward -  breaking ties with random choice
        arm = np.random.choice(np.flatnonzero(
            np.array(curr_arms) == np.array(curr_arms).max()))
        # Get the state of the arm
        arm_state = states[arm]
        # Get the reward obtained by the pulled arm
        rwd = bernoulli.rvs(p=M[arm, states[arm] - 1])
        # Sum the reward obtained to the total rwd of the algorithm
        tot_rwd_OG += rwd
        # Append the rwd to the individual rwds collected by the algorithm
        plot_rwds_OG[t] = plot_rwds_OG[t - 1] + rwd
        # Collect the sequence of pulls computed by the Greedy algorithm
        solution_OG.append(arm)
        # Update the states of the arms
        states = transition(states, arm)
    # print("Solution: ", solution_OG)
    # print("Total reward: ", tot_rwd_OG)

    # -------------------------- cumulative rewards ---------------------------
    print("Total reward ISI: ", tot_rwd_LSD)
    print("Total reward CSB: ", tot_rwd_CSB)
    print("Total reward OG : ", tot_rwd_OG)

    # -------------------------- plot comparison -----------------------------
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    d = d - 1
    ax = plt.figure(1, figsize=(12, 8))
    ax = plt.subplot(121)
    # to better show the rewards difference
    M_plot = M_true.copy()
    M_plot[1, 0] += 0.025
    ax = plt.axvline(x=10, ls="--", color="k")
    ax = plt.plot(np.arange(1, 12), M_plot[1, :12], marker="o",
                  label=r"$\mu_1(\tau)$", color="red")
    ax = plt.plot(np.arange(1, 12), M_plot[0, :12], marker="s",
                  label=r"$\mu_2(\tau)$", color="green")

    ax = plt.text(8.2, 0.21, "d=10", fontsize=16)
    x = np.arange(1, 12)
    labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    ax = plt.xticks(x, labels)
    ax = plt.tick_params(labelsize=16)
    ax = plt.legend(fontsize='xx-large')
    plt.grid()

    ax = plt.subplot(122)
    time_ax = np.arange(T)
    ax = plt.plot(time_ax, plot_rwds_LSD, label="ISI-CombUCB1")
    ax = plt. plot(time_ax, plot_rwds_CSB, label="CombUCB1")
    ax = plt.plot(time_ax, plot_rwds_OG, label="Oracle Greedy")
    ax = plt.grid()
    ax = plt.title("Cumulative Reward")
    ax = plt.xlabel('')
    ax = plt.ylabel('')
    ax = plt.legend(fontsize="xx-large")
    ax = plt.grid()
    name_fig = "res/runexp_trickOG.pdf"
    ax = plt.grid()
    ax = plt.savefig(name_fig, format="pdf", bbox_inches='tight')
    plt.show()

    with open('res/data_trickOG' + str(seed) + '.pkl', 'wb') as f:
        pickle.dump([seed, T, k, d, M_true, tot_rwd_LSD, tot_rwd_CSB,
                     tot_rwd_OG, plot_rwds_LSD, plot_rwds_CSB,
                     plot_rwds_OG], f)
