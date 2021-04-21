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
  * @file test_dist_fleet_dygraph_loss_consistent.py
  * @author liyang109@baidu.com
  * @date 2021-02-03 18:49
  * @brief 
  *
  **************************************************************************/
"""
from __future__ import print_function
import subprocess
import pytest


def test_dist_fleet_dygraph_loss_consistent_fleetrun():
    """test_dist_fleet_dygraph_loss_consistent_fleetrun"""
    p = subprocess.Popen(
                    "fleetrun --gpus=0,1 dist_fleet_dygraph_loss.py",
                    shell=True,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE)
    p.communicate()


def test_dist_fleet_dygraph_loss_consistent_launch():
    """test_dist_fleet_dygraph_loss_consistent_launch"""
    p = subprocess.Popen(
                    "python -m paddle.distributed.launch --gpus=0,1 dist_fleet_dygraph_loss.py",
                    shell=True,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE)
    p.communicate()

def test_dist_fleet_dygraph_spawn():
    """test_dist_fleet_dygraph_spawn."""
    p = subprocess.Popen(
        "python dist_fleet_dygraph_spawn.py",
        shell=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE)
    p.communicate()
