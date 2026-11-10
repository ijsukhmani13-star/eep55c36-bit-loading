"""Discrete bit loading for DMT (EEP55C36 final project, Track D).

Modules
-------
common    : input validation and the result container shared by both methods.
baseline  : greedy marginal-energy allocation, linear scan selection (the reproduction).
modified  : the same greedy rule with priority-queue (binary heap) selection (the change).
data      : synthetic multipath channel generator (ground truth and calibrated estimates).
metrics   : energy, feasibility, agreement and paired-statistics helpers.
"""
