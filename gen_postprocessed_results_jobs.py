"""Run the jobs that generate TEM Results.

Example commands:
python gen_postprocessed_results_jobs.py
"""
from copy import deepcopy
import json
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

fixed = json.load(open('/checkpoint/cinjon/spaceofmotion/bsn/peminf/fixeddata.json', 'r'))
# print('Fixed: ', len(fixed), fixed.keys())
# print(sorted(fixed.keys()))
fixed = {int(k.split('.')[0]): {i:j for i, j in v.items() if not i.startswith('base') and not 'curr_' in i and not 'start_time' in i} for k, v in fixed.items()}
# print(sorted(fixed.items())[0])

check = 0
for pem_results_subdir in os.listdir(pem_results_dir):
    c1, c2 = regex.match(pem_results_subdir).groups()
    c1 = int(c1)
    c2 = int(c2)
    counter = c2

    print(pem_results_subdir, c1, c2)
    if c1 in fixed:
        print('Got from fixed')
        job = fixed[c1]
    elif str(c1) in fixed:
        print('Got from fixed str')
        job = fixed[str(c1)]
    else:
        print('Run...')
        job = pemrun(find_counter=counter)

    print(sorted(job.items()))
    # job = pemrun(find_counter=counter)
    job['do_eval_after_postprocessing'] = True
    job['num_gpus'] = num_gpus
    job['num_cpus'] = 48
    job['gb'] = 64
    job['time'] = 1
    job['module'] = 'Post_processing'
    
    name = job['name']
    for ckpt_subdir in os.listdir(os.path.join(pem_results_dir, pem_results_subdir)):
        for postproc_width_init in [500, 300]:
            _job = deepcopy(job)
            _job['postproc_width_init'] = postproc_width_init
            dirkey = '%s/%s' % (pem_results_subdir, ckpt_subdir)
            _job['postprocessed_results_dir'] = os.path.join(postprocessed_results_dir, dirkey)
            _job['pem_inference_results_dir'] = os.path.join(pem_results_dir, dirkey)
            if 'thumos' in _job['dataset']:
                _job['video_info'] = _job['video_info'].replace('Full_Annotation.csv', 'thumos14_test_groundtruth.csv')
            elif 'gymnastics' in _job['dataset']:
                # This is actually using val.
                _job['video_info'] = _job['video_info'].replace('Full_Annotation.csv', 'ground_truth.csv')
            elif 'activitynet' in _job['dataset']:
                _job['video_info'] = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/activitynet_annotations/ground_truth_val.csv'
            _job['name'] = '%s.%s.width%d' % (pem_results_subdir, ckpt_subdir, postproc_width_init)

            func(_job, counter, email, code_directory)
            print()
            check += 1
print(check)
