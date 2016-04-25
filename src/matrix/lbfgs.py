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
def LbfgsOptions(lbfgs_config_str):
    parser = argparse.ArgumentParser(description"""
      These are the options for L-BFGS implementation.
      It pushes responsibility for determining when to stop, onto the user.
      This does not implement constrained L-BFGS, but it will
      handle constrained problems correctly as long as the function approaches
      +infinity (or -infinity for maximization problems) when it gets close to the
      bound of the constraint.  In these types of problems, you just let the
      function value be +infinity for minimization problems, or -infinity for
      maximization problems, outside these bounds).""",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stored-vecs-num", type=int, dafualt=10,
                        help="The number of stored vectors L-BFGS keeps.")
    parser.add_argument("--minimize", type=str, default = "true", choices = ["true", "false"], 
                        help="If true, we are minimizing, else maximizing.")
    parser.add_argument("--first-step-learning-rate", type=float, 
                        help="The very first step of L-BFGS is like gradient descent."
                        "If you want to configure the size of that step,"
                        "you can do it using this variable.", default = 1.0)
    # TODO  add other necessary configs

    lbfgs_args.append(parser.parse_args(shlex.split(lbfgs_config_str))

    return lbfgs_args

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


