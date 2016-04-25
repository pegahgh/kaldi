#!/usr/bin/env python

# Copyright 2016 Ke Li
#           2016 Pegah Ghahremani

import os
import subprocess
import argparse
import sys
import pprint
import logging
import imp
import traceback
import shutil
import math

lbfgs_lib = imp.load_source('ntl', 'lbfgs.py')


logger = loggin.getLogger(__name__)
logger.SetLevel(logging.INFO)
handler = logging.StreamHandler()
handler.SetLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Testing lbfgs libraray (lbfgs.py)')

def Main():
  try:
    lbfgs_opts = LbfgsOptions(true)
    init_x = 0;
    Lbfgs optimizer(init_x, lbfgs_opts)

  except Exception as e:
    
