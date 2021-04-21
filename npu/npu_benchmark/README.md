| API | 输入shape |Ascend FP16|V100 FP16 |AscendFP32| V100FP32|
  |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
  | elementwise_add | 1024  8192 | 282 | 266 | 166 | 265 |
  | matmul |1024 512 、512 1024  | - | 5.5 | 6.1 | 7.3 |
  | mul | 8192 3072 、3072 768 | 63 | 135 | 211 | 292 |
  | elementwise_mul | 1 、1232,19000 | 405 | 431 | 206 | 394 |
  | square | 3072 768 | 10 | 7 | 13 | 11 |
  | sqrt | 3072 768 | 10 | 9 | 16 | 11 |
  | scale | 16 512 512 | 13 | 10 | 20 | 12 |
  | slice | 16,1,768 | 5.9 | 4 | 6 | 4 |

  
  
性能指标为OP平均一次的时间；单位：us

结论：

- mul、elementwise_mul优于V100
- elementwise_add、square、sqrt、scale、slice性能劣于V100
- matmul op在npu上跑dtype=float16报错
