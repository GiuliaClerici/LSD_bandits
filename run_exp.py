import numpy as np
from data_tools import gen_M_trickCSBOG
from algo_tools import (get_Mx_rwd, oracle, reward_seq_XY_CombUCB1,
                        update_UCBs, transition, reward_seq_XY_firstpulls)
import matplotlib.pyplot as plt
import matplotlib
import pickle
from scipy.stats import bernoulli
import sys

# Checking the command line arguments - if there is one
# (we expect "cs" - calibration sequence), we store it and use it as a flag
# to indicate that we want to test also the calibration sequence approaches.
# Otherwise, the variable is store to indicate we do not want to test
# the calibration sequence approaches.
if len(sys.argv) == 2:
    calib_seq = sys.argv[1]
elif len(sys.argv) < 2:
    calib_seq = "no_cs"

# Initialize the parameters of the LSD bandit problem we are testing
k = 5  # Number of actions
pos = 3  # Position of the spike of action 1
size_M = 10  # Specify the length of the matrix
M_true = gen_M_trickCSBOG(k, size_M, pos)  # Generating the reward matrix
# M_true is the matrix storing the expected average reward
# for each action and state
print("M: ", M_true)
T = 3600  # Horizon - originally 5112
# We make sure that the horizon is divisible for the size of the block
# of every algorithm we are going to test

# Testing each algorithm on 10 runs, each one with a different random seed
n_runs = 1  # Number of runs - originally 10
for seed in range(n_runs):
    np.random.seed(seed)

# --------------------------------- Run CombUCB1 ------------------------------
    print("Running Vanilla CombUCB1...")
    d = 3  # Size of the block for vanilla CombUCB1
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
        # Generate the UCB est. matrix for representation of F
        UCBs_x = get_Mx_rwd(UCBs[:, 0:d], states, d)

        # Run the solver of the LP program
        seq = oracle(k, d, states, UCBs, UCBs_x)
        # Compute the rewards of arms pulled
        individual_rwds, _ = reward_seq_XY_CombUCB1(
            seq, states, M_true)
        # Sum the rewards obtained to the total rwd of the algorithm
        tot_rwd_CSB += individual_rwds.sum()

        # For each action we play
        for i in range(d):
            arm = seq[i]
            arm_state = states[arm]  # Select its state
            # Append the rwd to the individual rwds collected by the algorithm
            plot_rwds_CSB[(t * d) + i] = plot_rwds_CSB[(t * d) + i - 1] + \
                individual_rwds[i]
            # If the state of the arm is greater than d, then we are going to
            # update the estimate for arm played with state d, mu_i(d)
            if arm_state > d:
                arm_state = d
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
        # Generate the UCB est. matrix for representation of F -
        # all zeros since we don't consider the rewards of the first pulls
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

    # --------------------------------- Run Greedy ---------------------------
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
        arm = np.random.choice(
            np.flatnonzero(np.array(curr_arms) == np.array(curr_arms).max()))
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

    # ------------------------ Consider the two calibration sequence approaches
    if calib_seq == "cs":
        # # -------------------------- Calibration Sequence sigma 1 -----------
        print("Running the (worst) Calibration Sequence Approach...")
        d = 3  # Size of the block that will follow the calibration sequence
        cs_worst = np.arange(k)  # The worst possible calibration sequence
        states = np.ones((k), dtype=int)  # Initial states of the actions
        # All arms start with initial state at 1
        s = np.zeros((k, d + k), dtype=int)  # Matrix where the entries
        # indicate the number of times an action mu_i(j) has been played
        rwds_mean = np.zeros((k, d + k))  # Average reward of each
        # arm-state mu_i(j)
        solution_CS1 = []  # Array storing all the pulls CS1 plays over the
        # horizon
        tot_rwd_CS1 = 0.0  # Cumulative reward of CS1 over the horizon
        plot_rwds_CS1 = np.zeros((T))  # Array storing the individual rewards
        # collected in each time step
        cs_rwd = 0.0  # Reward obtained by the calibration sequence

        init_val = d * (np.sqrt((1.5 * np.log(k * d)) / 1))  # Initial value
        # we use to intialize the matrix containing the UCB indexes
        UCBs = np.ones((k, d + k)) * init_val  # Matrix storing the UCB indexes
        # for each mu_i(j)

        for t in range(T // (d + k)):
            # Get the rewards obtained by the calibration sequence sequence
            cs_individual_rwds, first_pulls = reward_seq_XY_CombUCB1(
                cs_worst, states, M_true)
            # Get the sum of the rewards obatined by the calibration sequence
            cs_rwd = cs_individual_rwds.sum()

            for j in range(k):
                # Append the rwd to the individual rwds collected by the alg
                plot_rwds_CS1[t * (d + k) + j] = (plot_rwds_CS1[
                                                  t * (d + k) + j - 1]
                                                  + cs_individual_rwds[j])
                # Compute the states of the arms after playing the calib. seq.
                states[j] = k - j
            # Get the Mx after playing the calib. seq.
            UCBs_x = get_Mx_rwd(UCBs, states, d + k)
            seq = oracle(k, d, states, UCBs[:, 0:d], UCBs_x[:, 0:d])
            seq_cs = np.append(cs_worst, seq)

            # Compute the rewards of arms pulled
            individual_rwds, first_pulls = reward_seq_XY_CombUCB1(
                seq, states, M_true)
            # Sum the rewards obtained to the total rwd of the algorithm
            tot_rwd_CS1 += cs_rwd + individual_rwds.sum()

            for i in range(d):
                # Append the rwd to the individual rwds collected by the alg
                plot_rwds_CS1[t * (d + k) + k + i] = (plot_rwds_CS1[
                                                      t * (d + k) + k + i - 1]
                                                      + individual_rwds[i])
                arm = seq[i]
                arm_state = states[arm]  # Get the state of the arm
                # For the action pulled, update the count of number of
                # plays for this action
                s[arm, arm_state - 1] += 1
                # Update the average reward of the arms
                if s[arm, arm_state - 1] > 1:
                    rwds_mean[arm, arm_state - 1] = (
                        (s[arm, arm_state - 1] - 1)
                        * rwds_mean[arm, arm_state - 1]
                        + individual_rwds[i]) / s[arm, arm_state - 1]
                else:
                    rwds_mean[arm, arm_state - 1] = individual_rwds[i]
                # Update the UCB indexes
                UCBs = update_UCBs(UCBs, k, d + k, rwds_mean, states,
                                   s, (t * (d + k)) + i + 1)
                # Update the states of the actions
                states = transition(states, arm)
            # Collect the sequence of pulls computed by the CS1 algorithm
            solution_CS1.append(seq_cs)
        # print("Solution: ", solution_CS1)
        # print("Total reward: ", tot_rwd_CS1)

        # -------------------------- Calibration Sequence sigma 2 ------------
        print("Running the (best) Calibration Sequence approach...")
        d = 3  # Size of the block that will follow the calibration sequence
        cs_best = np.flip(np.arange(k))  # The best possible calibration seq.
        states = np.ones((k), dtype=int)  # Initial states of the actions
        # All arms start with initial state at 1
        s = np.zeros((k, d + k), dtype=int)  # Matrix where the entries
        # indicate the number of times an action mu_i(j) has been played
        rwds_mean = np.zeros((k, d + k))  # Average reward of each
        # arm-state mu_i(j)
        solution_CS2 = []  # Array storing all the pulls CS2 plays over the
        # horizon
        tot_rwd_CS2 = 0.0  # Cumulative reward of CS2 over the horizon
        plot_rwds_CS2 = np.zeros((T))  # Array storing the individual rewards
        # collected in each time step
        cs_rwd = 0.0  # Reward obtained by the calibration sequence

        init_val = d * (np.sqrt((1.5 * np.log(k * d)) / 1))  # Initial value
        # we use to intialize the matrix containing the UCB indexes
        UCBs = np.ones((k, d + k)) * init_val  # Matrix storing the UCB indexes
        # for each mu_i(j)

        for t in range(T // (d + k)):
            # Get the rewards obtained by the calibration sequence sequence
            cs_individual_rwds, first_pulls = reward_seq_XY_CombUCB1(
                cs_best, states, M_true)
            # Get the sum of the rewards obatined by the calibration sequence
            cs_rwd = cs_individual_rwds.sum()

            for j in range(k):
                # Append the rwd to the individual rwds collected by the alg
                plot_rwds_CS2[t * (d + k) + j] = (plot_rwds_CS2[
                                                  t * (d + k) + j - 1]
                                                  + cs_individual_rwds[j])
                # Compute the states of the arms after playing the calib. seq.
                states[j] = j + 1
            # Get the Mx after playing the calib. seq.
            UCBs_x = get_Mx_rwd(UCBs, states, d + k)
            seq = oracle(k, d, states, UCBs[:, 0:d], UCBs_x[:, 0:d])
            seq_cs = np.append(cs_best, seq)

            # Compute the rewards of arms pulled
            individual_rwds, first_pulls = reward_seq_XY_CombUCB1(
                seq, states, M_true)
            # Sum the rewards obtained to the total rwd of the algorithm
            tot_rwd_CS2 += cs_rwd + individual_rwds.sum()

            for i in range(d):
                # Append the rwd to the individual rwds collected by the alg
                plot_rwds_CS2[t * (d + k) + k + i] = (plot_rwds_CS2[
                                                      t * (d + k) + k + i - 1]
                                                      + individual_rwds[i])
                arm = seq[i]
                arm_state = states[arm]  # Get the state of the arm
                # For the action pulled, update the count of number of
                # plays for this action
                s[arm, arm_state - 1] += 1
                # Update the average reward of the arms
                if s[arm, arm_state - 1] > 1:
                    rwds_mean[arm, arm_state - 1] = (
                        (s[arm, arm_state - 1] - 1)
                        * rwds_mean[arm, arm_state - 1]
                        + individual_rwds[i]) / s[arm, arm_state - 1]
                else:
                    rwds_mean[arm, arm_state - 1] = individual_rwds[i]
                # Update the UCB indexes
                UCBs = update_UCBs(UCBs, k, d + k, rwds_mean, states, s,
                                   (t * (d + k)) + i + 1)
                # Update the states of the actions
                states = transition(states, arm)
            # Collect the sequence of pulls computed by the CS2 algorithm
            solution_CS2.append(seq_cs)
        # print("Solution: ", solution_CS2)
        # print("Total reward: ", tot_rwd_CS2)

    # -------------------------- cumulative rewards ---------------------------

    print("Total reward ISI: ", tot_rwd_LSD)
    print("Total reward CSB: ", tot_rwd_CSB)
    print("Total reward OG : ", tot_rwd_OG)
    if calib_seq == "cs":
        print("Total reward CS1: ", tot_rwd_CS1)
        print("Total reward CS2: ", tot_rwd_CS2)

    # -------------------------- plot comparison -----------------------------
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    ax = plt.figure(1, figsize=(12, 8))
    ax = plt.subplot(121)
    # to better show the rewards difference
    M_plot = M_true.copy()
    M_plot[2, :8] -= 0.02
    M_plot[2, 5] += 0.03
    M_plot[3, :] -= 0.005
    ax = plt.axvline(x=3, ls="--", color="k")
    ax = plt.plot(np.arange(1, 11), M_plot[1, :], marker="o",
                  label=r"$\mu_1(\tau)$", color="red")
    ax = plt.plot(np.arange(1, 11), M_plot[2, :], marker="s",
                  label=r"$\mu_2(\tau)$", color="green")
    ax = plt.plot(np.arange(1, 11), M_plot[3, :], marker="d",
                  label=r"$\mu_{3, 4, 5}(\tau)$", color="blue")

    x = np.arange(1, 11)
    labels = ['1', '2', 'd=3', '4', '5', '6', '7', '8', '9', '10']
    ax = plt.xticks(x, labels)
    ax = plt.tick_params(labelsize=14)
    ax = plt.legend(fontsize='xx-large')
    plt.grid()

    ax = plt.subplot(122)
    time_ax = np.arange(0, T)
    ax = plt.plot(time_ax, plot_rwds_LSD, label="ISI-CombUCB1")
    ax = plt. plot(time_ax, plot_rwds_CSB, label="CombUCB1")
    ax = plt.plot(time_ax, plot_rwds_OG, label="Oracle Greedy")
    if calib_seq == "cs":
        ax = plt.plot(time_ax, plot_rwds_CS1, label="CS-worst")
        ax = plt.plot(time_ax, plot_rwds_CS2, label="CS-best")
    ax = plt.grid()
    ax = plt.title("Cumulative Reward")
    ax = plt.xlabel('')
    ax = plt.ylabel('')
    ax = plt.legend(fontsize="xx-large")
    ax = plt.grid()

    if calib_seq == "cs":
        name_fig = "res/runexp_trickboth_calibseq.pdf"
    else:
        name_fig = "res/runexp_trickboth.pdf"
    ax = plt.grid()
    ax = plt.savefig(name_fig, format="pdf", bbox_inches='tight')
    plt.show()

    if calib_seq == "cs":
        with open('res/data_trickboth_calib' + str(seed) + '.pkl', 'wb') as f:
            pickle.dump([seed, T, k, d, M_true, tot_rwd_LSD, tot_rwd_CSB,
                         tot_rwd_OG, tot_rwd_CS1, tot_rwd_CS2, plot_rwds_LSD,
                         plot_rwds_CSB, plot_rwds_OG, plot_rwds_CS1,
                         plot_rwds_CS2], f)
    else:
        with open('res/data_trickboth' + str(seed) + '.pkl', 'wb') as f:
            pickle.dump([seed, T, k, d, M_true, tot_rwd_LSD, tot_rwd_CSB,
                         tot_rwd_OG, plot_rwds_LSD, plot_rwds_CSB,
                         plot_rwds_OG], f)
