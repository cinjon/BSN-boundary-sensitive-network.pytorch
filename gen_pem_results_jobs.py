"""Run the jobs that generate TEM Results.

Example commands:
python gen_tem_results_jobs.py
"""
import os
import re
import sys

from run_on_cluster import fb_run_batch
from pem_jobs import run

email = 'cinjon@nyu.edu'
code_directory = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch'
anno_directory = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/gymnastics_annotations'
base_dir = '/checkpoint/cinjon/spaceofmotion/bsn'
pem_dir = os.path.join(base_dir, 'peminf')
pem_results_dir = os.path.join(pem_dir, 'results')
if not os.path.exists(pem_results_dir):
    os.makedirs(pem_results_dir)
ckpt_directory = os.path.join(pem_dir, 'do_ckpts')

regex = re.compile('.*(\d{5}).*')

for ckpt_subdir in os.listdir(ckpt_directory):
    counter = int(regex.match(ckpt_subdir).groups()[0])
    _job = run(find_counter=counter)
    _job['num_gpus'] = 8
    _job['num_cpus'] = 8 * 10
    _job['gb'] = 64 * 8
    _job['time'] = 4 # what time should this be?
    _job['pem_inference_results_dir'] = pem_results_dir
    _job['pem_results_subset'] = _job['pem_train_subset']
    _job['pem_inference_subset'] = 'full'
    _job['mode'] = 'inference'
    _job['checkpoint_path'] = os.path.join(ckpt_directory, ckpt_subdir)
    
    print(counter, sorted(_job.items()))
    fb_run_batch(_job, counter, email, code_directory)
