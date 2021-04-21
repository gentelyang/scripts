#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
#======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
#======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file test_dist_fleet_dygraph_gloo.py
  * @author liyang109@baidu.com
  * @date 2020-11-18 10:50
  * @brief 
  *
  **************************************************************************/
"""
from __future__ import print_function
import nose.tools as tools
import subprocess


def test_dygraph_gloo_init_rank0():
    """dygraph gloo init with ranks=2, local_rank=0."""
    p = subprocess.Popen(
                    "fleetrun --gpus=0,1 dist_fleet_dygraph_gloo.py",
                    shell=True,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE)
    p.communicate()




