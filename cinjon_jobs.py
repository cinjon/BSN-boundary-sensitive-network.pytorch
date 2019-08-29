"""Run the jobs in this file.

Example commands:
python cinjon_jobs.fb cims batch: Run the func jobs on fb using sbatch.
"""
import os
import sys

from run_on_cluster import fb_run_batch

email = 'cinjon@nyu.edu'
code_directory = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch'

func = fb_run_batch

counter = 0

anno_directory = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/gymnastics_annotations'
job = {
    'name': '2019.8.28.bsn',
    'video_anno': os.path.join(anno_directory, 'anno_fps12.on.json'),
    'video_info': os.path.join(anno_directory, 'video_info_new.csv'),
    'do_representation': True,
    'module': 'TEM',
    'time': 16,
}       
for num_gpus in [8, 4]:
    for lr in [1e-3, 1e-2, 3e-2]:
        for nvf in [25]:
            for svf in [6, 10, 14]:
                for num_run in range(2):
                    if num_gpus == 4 and (num_run > 0 or lr < 3e-2):
                        continue
                    
                    counter += 1
                    _job = {k: v for k, v in job.items()}
                    _job['num_gpus'] = num_gpus
                    _job['tem_training_lr'] = lr
                    _job['name'] = '%s-%d.%d' % (_job['name'], counter, num_run)
                    _job['num_cpus'] = num_gpus * 10
                    _job['gb'] = 64 * num_gpus
                    _job['num_videoframes'] = nvf
                    _job['skip_videoframes'] = svf
                    func(_job, counter, email, code_directory)
