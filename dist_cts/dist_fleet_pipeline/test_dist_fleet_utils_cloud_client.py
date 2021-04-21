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
  * @file test_dist_fleet_utils_cloud_client.py
  * @author liyang109@baidu.com
  * @date 2021-01-20 19:57
  * @brief 
  *
  **************************************************************************/
"""
import os
import sys
import subprocess


class TestFleetUtilsAfsApi():
    """TestFleetUtilsAfsApi"""
    def test_dist_fleet_utils_hdfs_client(self):
        """test_dist_fleet_worker"""
        cmd = 'fleetrun dist_fleet_utils_hdfs_client.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0

    def test_dist_fleet_utils_local_client(self):
        """test_dist_fleet_utils_local_client"""
        cmd = 'fleetrun dist_fleet_utils_localfs.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0