import os
import numpy as np

import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.types as types
import torch

#             pkl_file='/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/sep052019/full-data/train_nvvl.pkl',
        

class ActivityNetVideoPipe(Pipeline):
    def __init__(
            self, data, batch_size=16, num_threads=1, device_id=0, data, shuffle=True,
            seed=None, initial_prefetch_size=11, sequence_length=None,
    ):
        try:
            shard_id = torch.distributed.get_rank()
            num_shards = torch.distributed.get_world_size()
        except RuntimeError:
            shard_id = 0
            num_shards = 1
            
        super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=seed)
        self.input = ops.VideoReader(
            device="gpu", file_root=data, sequence_length=sequence_length, shard_id=0, 
            num_shards=1, random_shuffle=shuffle, initial_fill=initial_prefetch_size)

    def define_graph(self):
        output, labels = self.input(name="Reader")
        return output, labels


class ExternalInputIterator(object):
    def __init__(self, video_dir, annotations_dir, batch_size):
        self.video_dir = video_dir
        self.batch_size = batch_size
        with open(self.images_dir + "file_list.txt", 'r') as f:
            self.files = [line.rstrip() for line in f if line is not '']
        shuffle(self.files)
            
    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self
    
    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            jpeg_filename, label = self.files[self.i].split(' ')
            f = open(self.images_dir + jpeg_filename, 'rb')
            batch.append(np.frombuffer(f.read(), dtype = np.uint8))
            labels.append(np.array([label], dtype = np.uint8))
            self.i = (self.i + 1) % self.n
            return (batch, labels)
        
    next = __next__

                
def get_loader(args, pkl_root, phase):
    pkl_file=os.path.join(pkl_root, '%s_nvvl.pkl' % phase),
    sampler = None

    num_gpus = args.num_gpus
    batch_size_per_gpu = int(args.batch_size / num_gpus)
    num_threads_per_gpu = max(int(args.num_workers / num_gpus), 2)
    pipes = [
        VideoPipe(args, batch_size=batch_size_per_gpu, num_threads=num_threads_per_gpu, 
                  device_id=device_id, num_gpus=num_gpus) 
        for device_id in range(num_gpus)
    ]
    pipes[0].build()
    dali_iter = DALIGenericIterator(pipes, ['data', 'label'], pipes[0].epoch_size("Reader"))
    if phase == 'train':
        loader = NVVL(
            args.frames,
            args.is_cropped,
            args.crop_size,
            pkl_file=pkl_file,
            batchsize=args.batchsize,
            shuffle=True,
            distributed=True,
            device_id=args.rank % 8,
            fp16=args.fp16)
        num_batches = len(loader)
    elif phase == 'test':
        loader = NVVL(
            args.frames,
            args.is_cropped,
            args.crop_size,
            pkl_file=pkl_file,
            batchsize=args.batchsize,
            shuffle=True,
            distributed=True,
            device_id=args.rank % 8)
        num_batches = len(loader)    

    return loader, num_batches, sampler
