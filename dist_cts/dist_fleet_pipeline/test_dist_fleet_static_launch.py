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
  * @file test_dist_fleet_static_collective_launch.py
  * @author liyang109@baidu.com
  * @date 2020-11-17 17:31
  * @brief
  *
  **************************************************************************/
"""
from __future__ import print_function
import time
import signal
# import nose.tools as tools
import os
import subprocess
import time

single_data = [0.70575, 0.69835, 0.69342, 0.690098, 0.687781]

all_args = [
    "--gpus=0,1 --log_dir=mylog dist_fleet_static_fleetrun.py",
    "--gpus=0 --log_dir=mylog dist_fleet_static_fleetrun.py",
    "--gpus=0,1,2,3 --log_dir=mylog  dist_fleet_static_fleetrun.py",
    "--gpus=0,1 dist_fleet_static_fleetrun.py",
    "dist_fleet_static_fleetrun.py",
    "-log_dir=mylog dist_fleet_static_fleetrun.py",
    "--selected_gpus=0,1 --log_dir=mylog dist_fleet_static_fleetrun.py",
    "--selected_gpus=0 --log_dir=mylog dist_fleet_static_fleetrun.py",
    "--selected_gpus=0,1,2,3 --log_dir=mylog  dist_fleet_static_fleetrun.py",
    "--selected_gpus=0,1 dist_fleet_static_fleetrun.py",
        ]


class TestDistLaunch():
    """Test paddle.distributed.launch module cases."""

    # def __init__(self):
    #     self.single_data = [0.70575, 0.69835, 0.69342, 0.690098, 0.687781]
    #     self.test_info1 = []
    #     self.test_info2 = []
    #     all_args = [
    #         "--gpus=0,1 --log_dir=mylog dist_fleet_static_fleetrun.py",
    #         "--gpus=0 --log_dir=mylog dist_fleet_static_fleetrun.py",
    #         "--gpus=0,1,2,3 --log_dir=mylog  dist_fleet_static_fleetrun.py",
    #         "--gpus=0,1 dist_fleet_static_fleetrun.py",
    #         "dist_fleet_static_fleetrun.py",
    #         "-log_dir=mylog dist_fleet_static_fleetrun.py",
    #         "--selected_gpus=0,1 --log_dir=mylog dist_fleet_static_fleetrun.py",
    #         "--selected_gpus=0 --log_dir=mylog dist_fleet_static_fleetrun.py",
    #         "--selected_gpus=0,1,2,3 --log_dir=mylog  dist_fleet_static_fleetrun.py",
    #         "--selected_gpus=0,1 dist_fleet_static_fleetrun.py",
    #     ]

    # def check_data(self, loss, delta=None, expect=None):
    #     """
    #     ??????????????????.
    #     Args:
    #         loss (list): the loss will be checked.
    #         delta (float):
    #         expect (list):
    #     """
    #     if expect:
    #         expect_data = expect
    #     else:
    #         expect_data = single_data
    #     if delta:
    #         for i in range(len(expect_data)):
    #             tools.assert_almost_equal(loss[i], expect_data[i], delta=delta)
    #     else:
    #         for i in range(len(expect_data)):
    #             tools.assert_equal(loss[i], expect_data[i])

    def start_proc(self, cmd):
        """start process."""
        p = subprocess.Popen(
            "python -m paddle.distributed.launch --ips=127.0.0.1 " + cmd,
            shell=True,
            stderr=open("/tmp/launch.log", "wb"),
            stdout=subprocess.PIPE)
        p.communicate()
        with open('mylog/workerlog.0', 'r') as f:
            lines = f.readlines()[-1].lstrip('[').rstrip(']\n').split(',')
        loss = [eval(i) for i in lines]
        return loss

    def get_result(self, args):
        """get result"""
        test_info1 = []
        test_info2 = []
        loss1 = self.start_proc(args)
        time.sleep(2)
        loss2 = self.start_proc(args)
        test_info1.append(loss1)
        test_info2.append(loss2)
        assert len(test_info1[0]) == 5
        assert len(test_info2[0]) == 5
        # self.check_data(
        #     loss=test_info1[0], delta=3e-1, expect=test_info1[0])
        # self.check_data(
        #     loss=test_info2[0], delta=3e-1, expect=single_data)

    def test_dist_launch_Collective_2gpus_Tlog(self):
       """test_dist_launch_Collective_2gpus_Tlog."""
       args = all_args[0]
       self.get_result(args)

    def test_dist_launch_Collective_1gpus_Tlog(self):
        """test_dist_launch_Collective_1gpus_Tlog."""
        args = all_args[1]
        self.get_result(args)

    def test_dist_launch_Collective_4gpus_Tlog(self):
       """test_dist_launch_Collective_4gpus_Tlog."""
       args = all_args[2]
       self.get_result(args)


    def test_dist_launch_Collective_2gpus(self):
        """test_dist_launch_Collective_2gpus."""
        args = all_args[3]
        self.get_result(args)


    def test_dist_launch_Collective_default(self):
        """test_dist_launch_Collective_default."""
        args = all_args[4]
        self.get_result(args)


    def test_dist_launch_Collective_Tlog(self):
        """test_dist_launch_Collective_Tlog."""
        args = all_args[5]
        self.get_result(args)

    def test_dist_launch_Collective_2selected_gpus_log(self):
       """test_dist_launch_Collective_2selected_gpus_log."""
       args = all_args[6]
       self.get_result(args)

    def test_dist_launch_Collective_1selected_gpus_log(self):
        """test_dist_launch_Collective_1selected_gpus_log."""
        args = all_args[7]
        self.get_result(args)

    def test_dist_launch_Collective_4selected_gpus_log(self):
       """test_dist_launch_Collective_4selected_gpus_log."""
       args = all_args[8]
       self.get_result(args)


    def test_dist_launch_Collective_2selected_gpus(self):
        """test_dist_launch_Collective_2selected_gpus."""
        args = all_args[9]
        self.get_result(args)
