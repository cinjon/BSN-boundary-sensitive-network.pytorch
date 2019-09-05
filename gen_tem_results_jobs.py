"""Run the jobs that generate TEM Results.

Example commands:
python cinjon_jobs.py
"""
import os
import re
import sys

from run_on_cluster import fb_run_batch
from cinjon_jobs import run

email = 'cinjon@nyu.edu'
code_directory = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch'
anno_directory = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/gymnastics_annotations'
ckpt_directory = '/checkpoint/cinjon/spaceofmotion/bsn/do_ckpts'

regex = re.compile('.*(\d{5}).*')

for ckpt_subdir in os.listdir(ckpt_directory):
    counter = int(regex.match(ckpt_subdir).groups()[0])
    _job = run(find_counter=counter)
    _job['num_gpus'] = 8
    _job['num_cpus'] = 8 * 10
    _job['gb'] = 64 * 8
    _job['time'] = 4 # what time should this be?
    _job['tem_results_dir'] = '/checkpoint/cinjon/spaceofmotion/bsn/teminf'
    _job['tem_results_subset'] = _job['tem_train_subset']
    _job['mode'] = 'inference'
    _job['checkpoint_path'] = os.path.join(ckpt_directory, ckpt_subdir)
    print(counter, _job)
    fb_run_batch(_job, counter, email, code_directory)
