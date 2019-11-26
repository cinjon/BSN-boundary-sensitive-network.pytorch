"""Run the jobs that generate PGM features.

Example commands:
python gen_pgm_results_jobs.py
"""
from copy import deepcopy
import os
import re
import sys

from run_on_cluster import fb_run_batch
from tem_jobs import run

email = 'cinjon@nyu.edu'
code_directory = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch'
base_dir = '/checkpoint/cinjon/spaceofmotion/bsn'
tem_dir = os.path.join(base_dir, 'teminf')
tem_results_dir = os.path.join(tem_dir, 'results')
pgm_proposals_dir = os.path.join(base_dir, 'pgmprops')
pgm_feats_dir = os.path.join(base_dir, 'pgmfeats')

regex = re.compile('.*(\d{5}).*')

check = 0
for tem_results_subdir in os.listdir(tem_results_dir):
    counter = int(regex.match(tem_results_subdir).groups()[0])

    
    print(tem_results_dir, tem_results_subdir, counter)
    job = run(find_counter=counter)
    if type(job) == tuple:
        job = job[1]

    name = job['name']
    for ckpt_subdir in os.listdir(os.path.join(tem_results_dir, tem_results_subdir)):
        _job = deepcopy(job)

        if 'thumos' in _job['dataset']:
            _job['video_anno'] = os.path.join(_job['video_info'], 'thumos_anno_action.json')
        elif 'gymnastics' in _job['dataset']:
            _job['video_anno'] = os.path.join(_job['video_info'], 'gymnastics_anno_action.sep052019.json')
            
        _job['num_gpus'] = 0
        _job['num_cpus'] = 48
        _job['pgm_thread'] = 40
        _job['gb'] = 64
        _job['time'] = 3
        _job['tem_results_dir'] = os.path.join(tem_results_dir, tem_results_subdir, ckpt_subdir)
        
        propdir = os.path.join(pgm_proposals_dir, tem_results_subdir, ckpt_subdir)
        if not os.path.exists(propdir):
            os.makedirs(propdir)
        _job['pgm_proposals_dir'] = propdir

        featsdir = os.path.join(pgm_feats_dir, tem_results_subdir, ckpt_subdir)
        if not os.path.exists(featsdir):
            os.makedirs(featsdir)
        _job['pgm_features_dir'] = featsdir
            
        _job['module'] = 'PGM'
        _job['mode'] = 'pgm'
        _job['name'] = '%s.%s' % (name, ckpt_subdir)

        fb_run_batch(_job, counter, email, code_directory)
        check += 1
        print('\n')
print("Count: ", check)
