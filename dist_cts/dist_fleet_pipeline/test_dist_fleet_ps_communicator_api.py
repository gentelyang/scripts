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
  * @file test_dist_fleet_ps_communicator_api.py
  * @author liyang109@baidu.com
  * @date 2021-01-18 19:16
  * @brief 
  *
  **************************************************************************/
"""
import os
import subprocess


class TestPsUtileApi():
    """test all api"""
    def test_dist_fleet_ps_all_reduce(self):
        """test_dist_fleet_ps_all_reduce"""
        cmd = 'fleetrun dist_ps_all_reduce.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0

    def test_dist_fleet_ps_barrier(self):
        """test_dist_fleet_ps_barrier"""
        cmd = 'fleetrun dist_ps_barrier.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0

    def test_dist_fleet_ps_get_file_shard(self):
        """test_dist_fleet_ps_get_file_shard"""
        cmd = 'fleetrun dist_ps_get_file_shard.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0

    def test_dist_fleet_ps_print_on_rank(self):
        """test_dist_fleet_ps_print_on_rank"""
        cmd = 'fleetrun dist_ps_get_file_shard.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0