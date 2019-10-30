"""Run the jobs that generate TEM Results.

Example commands:
python gen_postprocessed_results_jobs.py
"""
from copy import deepcopy
import os
import re
import sys

from run_on_cluster import fb_run_batch as func
from pem_jobs import run as pemrun

email = 'cinjon@nyu.edu'
code_directory = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch'

base_dir = '/checkpoint/cinjon/spaceofmotion/bsn'
checkpoint_path = os.path.join(base_dir, 'checkpoint', 'pem')
pem_dir = os.path.join(base_dir, 'peminf')
pem_results_dir = os.path.join(pem_dir, 'results')
postprocessed_results_dir = os.path.join(base_dir, 'postprocessing')

regex = re.compile('.*(\d{5}).ckpt.*-(\d{5}).*')
num_gpus = 0

for pem_results_subdir in os.listdir(pem_results_dir):
    c1, c2 = regex.match(pem_results_subdir).groups()
    c1 = int(c1)
    c2 = int(c2)
    counter = c2
    # counter = int(regex.match(pem_results_subdir).groups()[0])
    job = pemrun(find_counter=counter)
    job['do_eval_after_postprocessing'] = True
    job['num_gpus'] = num_gpus
    job['num_cpus'] = 48
    job['gb'] = 64
    job['time'] = 1
    job['module'] = 'Post_processing'
    
    name = job['name']
    for ckpt_subdir in os.listdir(os.path.join(pem_results_dir, pem_results_subdir)):
        _job = deepcopy(job)
        dirkey = '%s/%s' % (pem_results_subdir, ckpt_subdir)
        _job['postprocessed_results_dir'] = os.path.join(postprocessed_results_dir, dirkey)
        _job['pem_inference_results_dir'] = os.path.join(pem_results_dir, dirkey)
        if 'thumos' in _job['dataset']:
            _job['video_info'] = _job['video_info'].replace('Full_Annotation.csv', 'thumos14_test_groundtruth.csv')
        elif 'gymnastics' in _job['dataset']:
            _job['video_info'] = _job['video_info'].replace('Full_Annotation.csv', 'ground_truth.csv')
        _job['name'] = '%s.%s' % (pem_results_subdir, ckpt_subdir)

        func(_job, counter, email, code_directory)
        print()
