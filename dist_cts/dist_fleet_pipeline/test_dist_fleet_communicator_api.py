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
  * @file test.py
  * @author liyang109@baidu.com
  * @date 2020-12-30 15:53
  * @brief
  *
  **************************************************************************/
"""
import os
import subprocess

os.system("export CUDA_VISIBLE_DEVICES=0,1")
class TestApi():
    """test all comm api"""
    def test_collective_all_gather(self):
        """test_collective_all_gather"""
        cmd = 'fleetrun dist_collective_all_gather.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_all_reduce_max(self):
        """test_collective_all_reduce_max"""
        cmd = 'fleetrun dist_collective_all_reduce_max.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_all_reduce_min(self):
        """test_collective_all_reduce_min"""
        cmd = 'fleetrun dist_collective_all_reduce_min.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_all_reduce_prod(self):
        """test_collective_all_reduce_prod"""
        cmd = 'fleetrun dist_collective_all_reduce_prod.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_all_reduce_sum(self):
        """test_collective_all_reduce_sum"""
        cmd = 'fleetrun dist_collective_all_reduce_sum.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_barrier(self):
        """test_collective_barrier"""
        cmd = 'fleetrun dist_collective_barrier.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_broadcast_c1(self):
        """test_collective_broadcast_c1"""
        cmd = 'fleetrun dist_collective_broadcast_c1.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_broadcast_c2(self):
        """test_collective_broadcast_c2"""
        cmd = 'fleetrun dist_collective_broadcast_c2.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_get_rank(self):
        """test_collective_get_rank"""
        cmd = 'fleetrun dist_collective_get_rank.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_get_world_size(self):
        """test_collective_get_world_size"""
        cmd = 'fleetrun dist_collective_get_world_size.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_get_current_endpoint(self):
        """test_collective_get_current_endpoint"""
        cmd = 'fleetrun dist_collective_parallelenv_get_current_endpoint.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_get_device_id(self):
        """test_collective_get_device_id"""
        cmd = 'fleetrun dist_collective_parallelenv_get_device_id.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_parallelenv_get_rank(self):
        """test_collective_parallelenv_get_rank"""
        cmd = 'fleetrun dist_collective_parallelenv_get_rank.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_parallelenv_get_trainer_endpoints(self):
        """test_collective_parallelenv_get_trainer_endpoints"""
        cmd = 'fleetrun dist_collective_parallelenv_get_trainer_endpoints.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_parallelenv_world_size(self):
        """test_collective_parallelenv_world_size"""
        cmd = 'fleetrun dist_collective_parallelenv_world_size.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_reduce_max(self):
        """test_collective_reduce_max"""
        cmd = 'fleetrun dist_collective_reduce_max.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_reduce_min(self):
        """test_collective_reduce_min"""
        cmd = 'fleetrun dist_collective_reduce_min.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_reduce_prod(self):
        """test_collective_reduce_prod"""
        cmd = 'fleetrun dist_collective_reduce_prod.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_reduce_sum(self):
        """test_collective_reduce_sum"""
        cmd = 'fleetrun dist_collective_reduce_sum.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_reduceop_sum(self):
        """test_collective_reduceop_sum"""
        cmd = 'fleetrun dist_collective_reduceop_sum.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_reduceop_max(self):
        """test_collective_reduceop_max"""
        cmd = 'fleetrun dist_collective_reduceop_max.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_reduceop_min(self):
        """test_collective_reduceop_min"""
        cmd = 'fleetrun dist_collective_reduceop_min.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_reduceop_prod(self):
        """test_collective_reduceop_prod"""
        cmd = 'fleetrun dist_collective_reduceop_prod.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_scatter(self):
        """test_scatter"""
        cmd = 'fleetrun dist_collective_scatter.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_barrier_worker(self):
        """test_collective_barrier_worker"""
        cmd = 'fleetrun dist_fleet_barrier_worker.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_dist_init_parallel_env(self):
        """test_dist_init_parallel_env"""
        cmd = 'fleetrun dist_init_parallel_env.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_dist_parallelenv(self):
        """test_dist_parallelenv"""
        cmd = 'fleetrun dist_collective_parallelenv_get_rank.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0

    def test_collective_send_recv(self):
        """test_send_recv"""
        cmd = 'fleetrun dist_collective_send_recv.py'
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0