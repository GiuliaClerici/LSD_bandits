# LSD_bandits

This repository contains the code to reproduce the experiments presented in the paper [_"Break your Bandit Routine with LSD Rewards: a Last Switch Dependent Analysis of Satiation and Seasonality"_](https://arxiv.org/abs/2110.11819) by Laforgue et al. (2021).


This repository contains the following files:
- [run_exp.py](run_exp.py) contains the code to reproduce the experiments presented in Section 4 and Figure 3 of the Supplementary Material.
- [run_exp_sup.py](run_exp_sup.py) contains the code to reproduce the experiment presented in Figure 4 of the Supplementary Material.
- [algo_tools.py](algo_tools.py) contains the functions called in the above scripts to implement the algorithms ISI-CombUCB1, CombUCB1, Oracle Greedy, and the two Calibration Sequence approaches.
- [data_tools.py](data_tools.py) contains the functions used to generate the instances of the Bernoulli LSD bandit we test.
- [res](res) is the folder where graphs and data are saved.

In order to run the experiment presented in Section 4 of the paper, run the following command (~1min):
```python
$ python run_exp.py
```
In order to run the same experiment but with the additional benchmark of Calibration Sequence approaches, run the following command (~1.5min):
```python
$ python run_exp.py cs
```
To run the additional experiment presented in Figure 4 of the Supplementary Material, run the following command (~2min):
```python
$ python run_exp_sup.py
```

The plots generated are saved in the [res](res) folder.
