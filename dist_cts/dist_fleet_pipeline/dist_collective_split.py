#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:gentelyang  time:2021-06-15
import paddle
from paddle.distributed import init_parallel_env

paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
init_parallel_env()
data = paddle.randint(0, 8, shape=[10,4])
emb_out = paddle.distributed.split(
    data,
    (8, 8),
    operation="embedding",
    num_partitions=2)
print(emb_out)