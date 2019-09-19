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

base_dir = '/checkpoint/cinjon/spaceofmotion/bsn'
pem_dir = os.path.join(base_dir, 'peminf')
pem_results_dir = os.path.join(pem_dir, 'results')
if not os.path.exists(pem_results_dir):
    os.makedirs(pem_results_dir)
ckpt_directory = os.path.join(pem_dir, 'do_ckpts')

regex = re.compile('.*ckpt.*-(\d{5}).*')
num_gpus = 4

for ckpt_subdir in os.listdir(ckpt_directory):
    counter = int(regex.match(ckpt_subdir).groups()[0])
    _job = run(find_counter=counter)
    _job['num_gpus'] = num_gpus
    _job['num_cpus'] = num_gpus * 10
    _job['gb'] = 64 * num_gpus
    _job['time'] = 1.5
    _job['pem_inference_results_dir'] = pem_results_dir
    _job['pem_inference_subset'] = 'full'
    _job['mode'] = 'inference'
    _job['checkpoint_path'] = os.path.join(ckpt_directory, ckpt_subdir)

    name = _job['name']
    for ckpt_epoch in [15, 30]:
        _job['checkpoint_epoch'] = ckpt_epoch
        _job['name'] = '%s.ckpt%d' % (name, ckpt_epoch)
        print(counter, _job['name'])
        print(sorted(_job.items()))
        fb_run_batch(_job, counter, email, code_directory)
