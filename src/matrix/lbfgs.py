#!/usr/bin/env python 

# Copyright 2016 Ke Li
#           2016 Pegah Ghahremani

import os
import argparse
import sys
import pprint
import logging
import imp
import traceback
import shutil
import math

# This is the option for an implementation of L-BFGS. 
# These are the options for L-BFGS implementation.
# It pushes responsibility for determining when to stop, onto the user.
# This does not implement constrained L-BFGS, but it will
# handle constrained problems correctly as long as the function approaches
# +infinity (or -infinity for maximization problems) when it gets close to the
# bound of the constraint.  In these types of problems, you just let the
# function value be +infinity for minimization problems, or -infinity for
# maximization problems, outside these bounds).""",
def LbfgsOptions(minimize):
  # TOD add desrciption for each paramters.
  lbfgs_opts.m = 10
  lbfgs_opts.minimize = minimize
  lbfgs_opts.first_step_learning_rate = 1.0
  lbfgs_opts.first_step_length = 0.0
  lbfgs_opts.first_step_impr = 0.0
  lbfgs_opts.c1 = 0.0001
  lbfgs_opts.c2 = 0.9
  lbfgs_opts.d = 2.0
  lbfgs_opts.max_line_search_iters = 50
  lbfgs_opts.avg_step_length = 4
  return lbfgs_opts

class Lbfgs():

  def __init__(self, init_x, lbfgs_opts):
    # TODO we need to initialize necessary parameters
    self.lbfgs_opts_ = lbfgs_opts
    self.k_ = 0
    self.computation_state_ = kBeforeStep
    self.H_was_set_ = False
    assert(self.lbfgs_opts.stored_vec_num > 0)
    assert(self.dim > 0)
    self.x_ = init_x  # This is the value of x_k
    self.new_x_ = init_x # This is where we'll evaluate the function next. 
    self.deriv_ = []
    self.deriv_.resize((dim, 1))
    self.temp_ = []
    self.temp_.resize((dim, 1))

  # This returns the value of the variable x that has the best
  # objective function so far, and the corresponding objective function value
  # This onlye be called only at the end. 
  def GetValue(self):
     return (self.best_x_, self.best_f_)


