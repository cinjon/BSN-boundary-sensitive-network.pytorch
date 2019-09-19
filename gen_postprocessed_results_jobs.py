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

regex = re.compile('.*ckpt.*-(\d{5}).*')
num_gpus = 0


for pem_results_subdir in os.listdir(pem_results_dir):
    counter = int(regex.match(pem_results_subdir).groups()[0])
    job = pemrun(find_counter=counter)
    
    name = job['name']
    for ckpt_subdir in os.listdir(os.path.join(pem_results_dir, pem_results_subdir)):
        _job = deepcopy(job)
        _job['module'] = 'Post_processing'
        dirkey = '%s/%s' % (pem_results_subdir, ckpt_subdir)
        _job['postprocessed_results_dir'] = os.path.join(postprocessed_results_dir, dirkey)
        _job['pem_inference_results_dir'] = os.path.join(pem_results_dir, dirkey)
        if 'thumos' in _job['dataset']:
            _job['video_info'] = _job['video_info'].replace('Full_Annotation.csv', 'thumos14_test_groundtruth.csv')
        _job['name'] = '2019.09.18.%s.%s' % (pem_results_subdir, ckpt_subdir)
        _job['num_gpus'] = num_gpus
        _job['num_cpus'] = 48
        _job['gb'] = 64
        _job['time'] = 4
            
        func(_job, counter, email, code_directory)
