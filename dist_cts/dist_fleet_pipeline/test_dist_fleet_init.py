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


os.system("export CUDA_VISIBLE_DEVICES=0,1")
class TestApi():
    """test all api"""
    def test_dist_fleet_init(self):
        """test_dist_fleet_init"""
        cmd = 'fleetrun dist_fleet_init.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_dist_fleet_init_paddlecloud_role(self):
        """test_dist_fleet_init_paddlecloud_role"""
        cmd = 'fleetrun dist_fleet_init_cloudrole.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_dist_fleet_init_collective_role(self):
        """test_dist_fleet_init_collective_role"""
        cmd = 'fleetrun dist_fleet_init_collective.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_dist_fleet_init_collective_strategy(self):
        """test_dist_fleet_init_collective_strategy"""
        cmd = 'fleetrun dist_fleet_init_strategy.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0