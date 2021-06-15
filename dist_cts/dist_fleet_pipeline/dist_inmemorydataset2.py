#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:gentelyang  time:2021-06-14
import paddle
paddle.enable_static()

dataset = paddle.distributed.InMemoryDataset()
dataset.init(
    batch_size=1,
    thread_num=2,
    input_type=1,
    pipe_command="cat",
    use_var=[])
dataset._init_distributed_settings(
    parse_ins_id=True,
    parse_content=True,
    fea_eval=True,
    candidate_size=10000)
assert dataset._init_distributed_settings is not None
dataset.update_settings(batch_size=2)
dataset.load_into_memory()
dataset.local_shuffle()
dataset.preload_into_memory()
dataset.wait_preload_done()