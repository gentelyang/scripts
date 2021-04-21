#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:gentelyang  time:2021-04-12
import sys
import numpy as np
import paddle
from paddle.distributed import init_parallel_env, ReduceOp


paddle.distributed.init_parallel_env()
d1 = np.array([1, 2, 3])
d2 = np.array([2, 3, 4])
tensor1 = paddle.to_tensor(d1)
tensor2 = paddle.to_tensor(d2)
gp = paddle.distributed.new_group([0,1])
print("test_new_group ...ok")
tmp = np.array([0, 0, 0])
result = paddle.to_tensor(tmp)


def test_new_group_scatter():
    paddle.distributed.scatter(
        result, [tensor2, tensor1],
        src=0,
        group=gp,
        use_calc_stream=True)
    if gp.rank == 0:
        assert np.array_equal(result, tensor2)
    elif gp.rank == 1:
        assert np.array_equal(result, tensor1)
    print("test_new_group_scatter... ok")

def test_new_group_reduce_sum():
    paddle.distributed.reduce(result, dst=0, group=gp, use_calc_stream=True)

    if gp.rank == 0:
        assert np.array_equal(result.numpy(),
                              [6 , 10, 14])
    elif gp.rank == 1:
        assert np.array_equal(result.numpy(), [6 , 10, 14])
    print("test_new_group_reduce... ok")

def test_new_group_allreduce_sum():
    paddle.distributed.all_reduce(result, use_calc_stream=True)
    assert np.array_equal(
        result.numpy(),
        [3, 5, 7])
    print("test_new_group_all_reduce api ok")

def test_new_group_wait():
    paddle.distributed.wait(result, gp, use_calc_stream=True)
    paddle.distributed.wait(result, gp, use_calc_stream=False)
    assert 1 == 1
    print("test_new_group_wait... ok")

def test_new_group_all_gather():
    result = []
    # paddle.distributed.all_gather(
    #     result, [self.tensor1, self.tensor1], group=gp, use_calc_stream=True)
    paddle.distributed.all_gather(
        result, tensor1, group=gp, use_calc_stream=True)
    assert np.array_equal(result[0], tensor1)
    assert np.array_equal(result[1], tensor1)
    print("test_new_group_all_gather... ok")

def test_new_group_broadcast():
    tmp = np.array([0, 0, 0])
    result = paddle.to_tensor(tmp)
    paddle.distributed.broadcast(
        result, src=1, group=gp, use_calc_stream=True)
    if gp.rank == 0:
        assert np.array_equal(result.numpy(), [0, 0, 0])
        print("test_new_group_broadcast_rank0... ok")
    elif gp.rank == 1:
        assert np.array_equal(result.numpy(), [0, 0, 0])
        print("test_new_group_broadcast_rank1... ok")

def test_new_group_barrier():
    paddle.distributed.barrier(group=gp)
    assert 1 == 1
    print("test_new_group_barrier... ok")


if __name__ == '__main__':
    test_new_group_scatter()
    test_new_group_all_gather()
    test_new_group_allreduce_sum()
    test_new_group_reduce_sum()
