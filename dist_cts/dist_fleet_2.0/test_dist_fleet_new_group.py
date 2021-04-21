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
  * @file test_dist_fleet_init.py
  * @author liyang109@baidu.com
  * @date 2021-01-18 16:16
  * @brief
  *
  **************************************************************************/
"""
import os
import subprocess
from utils import run_priority

os.system("export CUDA_VISIBLE_DEVICES=0,1")
class TestApi():
    """test all api"""
    def test_dist_fleet_dygraph_new_group(self):
        """test_dist_fleet_dygraph_new_group"""
        cmd = 'fleetrun --gpus 0,1 dist_fleet_dygraph_new_group1.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0

    def test_dist_fleet_dygraph_new_group(self):
        """test_dist_fleet_dygraph_new_group"""
        cmd = 'fleetrun --gpus 0,1 dist_fleet_dygraph_new_group2.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
