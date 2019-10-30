import os
import numpy as np

import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.types as types
import torch

import opts
        

class ActivityNetVideoPipe(Pipeline):
    def __init__(
            self, args, file_list, batch_size=16, num_threads=1, device_id=0, shuffle=False,
    ):
        """
        Args:
          step: The frame interval between each sequence.
          stride: The distance between consecutive frames in each sequence.
        """
        shard_id = 0
        num_shards = 1

        super(ActivityNetVideoPipe, self).__init__(batch_size, num_threads, device_id, seed=args['seed'])
        print(shuffle, args['dist_videoframes'], args['skip_videoframes'], args['num_videoframes'], file_list)
        self.input = ops.VideoReader(
            device="gpu", file_list=file_list, sequence_length=args['num_videoframes'], shard_id=shard_id, 
            num_shards=num_shards, random_shuffle=shuffle, initial_fill=args['initial_prefetch_size'],
            step=args['dist_videoframes'], stride=args['skip_videoframes'],
            skip_vfr_check=False
        )


    def define_graph(self):
        images, labels = self.input(name="Reader")
        return images.gpu(), labels.gpu() 

                
def get_loader(args, phase):
    file_list = args['train_video_file_list'] if phase == 'train' else None
    num_gpus = args['num_gpus']
    batch_size_per_gpu = int(args['tem_batch_size'] / num_gpus)
    num_threads_per_gpu = max(int(args['data_workers'] / num_gpus), 2)
    pipes = [
        ActivityNetVideoPipe(
            args, file_list, batch_size=batch_size_per_gpu, num_threads=num_threads_per_gpu, 
            device_id=device_id)
        for device_id in range(1) # num_gpus)
    ]
    pipes[0].build()
    epoch_size = pipes[0].epoch_size("Reader")
    dali_iter = DALIGenericIterator(pipes, ['data', 'label'], epoch_size)
    return dali_iter, epoch_size


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    dali_iter, epoch_size = get_loader(opt, 'train')
    print("Size? : ", epoch_size)
    from collections import defaultdict
    counts = defaultdict(int)
    for i, inputs in enumerate(dali_iter):
        print(i)
        for thread_input in inputs:
            data = thread_input['data']
            label = thread_input['label']
            for k in label:
                counts[k.item()] += 1
            print(counts)
            # print(data.shape)
            # print(label)
    print(counts)
