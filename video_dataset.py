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
            initial_prefetch_size=11, sequence_length=None, step=None,
            stride=None
    ):
        """
        Args:
          step: The frame interval between each sequence.
          stride: The distance between consecutive frames in each sequence.
        """
        shard_id = 0
        num_shards = 1

        super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=seed)
        self.input = ops.VideoReader(
            device="gpu", file_list=file_list, sequence_length=args.num_videoframes, shard_id=shard_id, 
            num_shards=num_shards, random_shuffle=shuffle, initial_fill=args.initial_prefetch_size,
            step=args.dist_videoframes, stride=args.skip_videoframes
        )
        # What DaliInterpType to use?
        self.resize = ops.Resize(
            device="cpu",
            resize_x=300,
            resize_y=300,
            min_filter=types.DALIInterpType.INTERP_TRIANGULAR) 

    def define_graph(self):
        images, labels = self.input(name="Reader")
        images = self.resize(images)
        return images.gpu(), labels.gpu()

                
def get_loader(args, phase):
    file_list = args.train_video_file_list if phase == 'train' else None
    num_gpus = args.num_gpus
    batch_size_per_gpu = int(args.batch_size / num_gpus)
    num_threads_per_gpu = max(int(args.num_workers / num_gpus), 2)
    pipes = [
        ActivityNetVideoPipe(args, file_list, batch_size=batch_size_per_gpu, num_threads=num_threads_per_gpu, 
                  device_id=device_id, num_gpus=num_gpus) 
        for device_id in range(num_gpus)
    ]
    pipes[0].build()
    dali_iter = DALIGenericIterator(pipes, ['data', 'label'], pipes[0].epoch_size("Reader"))
    return dali_iter


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    dali_iter = get_loader(opt, 'train')
