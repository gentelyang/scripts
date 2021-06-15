#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:gentelyang  time:2021-06-15
import numpy as np
import paddle
from utils import run_priority
from paddle.distributed import init_parallel_env

types = [np.float16, np.float32, np.float64, np.int32, np.int64]

@run_priority(level='P0')
def test_send1():
    """test_send1"""
    init_parallel_env()
    if paddle.distributed.ParallelEnv().rank == 0:
        data = paddle.to_tensor([7, 8, 9])
        paddle.distributed.send(data, dst=1)
    else:
        data = paddle.to_tensor([1, 2, 3])
        paddle.distributed.recv(data, src=0)
    out = data.numpy()
    assert out[0] == 7
    assert out[1] == 8
    assert out[2] == 9

@run_priority(level='P0')
def test_send2():
    """test_send2"""
    init_parallel_env()
    if paddle.distributed.ParallelEnv().rank == 0:
        data = paddle.to_tensor([7, 8, 9])
        paddle.distributed.send(data, dst=1, group=None, use_calc_stream=True)
    else:
        data = paddle.to_tensor([1, 2, 3])
        paddle.distributed.recv(data, src=0)
    out = data.numpy()
    assert out[0] == 7
    assert out[1] == 8
    assert out[2] == 9

@run_priority(level='P0')
def test_send3():
    """test_send3"""
    init_parallel_env()
    if paddle.distributed.ParallelEnv().rank == 0:
        data = paddle.to_tensor([7, 8, 9])
        paddle.distributed.send(data, dst=1, group=None, use_calc_stream=False)
    else:
        data = paddle.to_tensor([1, 2, 3])
        paddle.distributed.recv(data, src=0)
    out = data.numpy()
    assert out[0] == 7
    assert out[1] == 8
    assert out[2] == 9

@run_priority(level='P0')
def test_send3():
    """test_send3"""
    init_parallel_env()
    if paddle.distributed.ParallelEnv().rank == 0:
        data = paddle.to_tensor([7, 8, 9])
        paddle.distributed.send(data, dst=1, group=None, use_calc_stream=False)
    else:
        data = paddle.to_tensor([1, 2, 3])
        paddle.distributed.recv(data, src=0)
    out = data.numpy()
    assert out[0] == 7
    assert out[1] == 8
    assert out[2] == 9