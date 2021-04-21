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
  * @file dist_ps_barrier.py
  * @author liyang109@baidu.com
  * @date 2021-01-17 14:06
  * @brief 
  *
  **************************************************************************/
"""
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet import PaddleCloudRoleMaker
import sys
import os
from dist_utils import run_priority

os.environ["PADDLE_WITH_GLOO"] = "2"


@run_priority(level='P0')
def test_barrier():
    """test ps barrier"""
    role = PaddleCloudRoleMaker(
        is_collective=False,
        init_gloo=True,
        path="./tmp_gloo")
    fleet.init(role)

    if fleet.is_server():
        fleet.util.barrier("server")
        print("test_all_servers barrier ... ok")
    elif fleet.is_worker():
        fleet.util.barrier("worker")
        print("test_all_workers barrier ... ok")
    fleet.util.barrier("test_barrier ... ok")
    print("all servers and workers arrive here")

if __name__ == "__main__":
    test_barrier()