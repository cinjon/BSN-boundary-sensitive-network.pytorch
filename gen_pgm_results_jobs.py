"""Run the jobs that generate PGM features.

Example commands:
python gen_pgm_results_jobs.py
"""
import os
import re
import sys

from run_on_cluster import fb_run_batch
from tem_jobs import run

email = 'cinjon@nyu.edu'
code_directory = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch'
anno_directory = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/gymnastics_annotations'
base_dir = '/checkpoint/cinjon/spaceofmotion/bsn'
tem_dir = os.path.join(base_dir, 'teminf')
tem_results_dir = os.path.join(tem_dir, 'results')
pgm_proposals_dir = os.path.join(base_dir, 'pgmprops')
pgm_feats_dir = os.path.join(base_dir, 'pgmfeats')

regex = re.compile('.*(\d{5}).*')

for tem_results_subdir in os.listdir(tem_results_dir):
    counter = int(regex.match(tem_results_subdir).groups()[0])
    if counter != 101:
        continue
    
    _job = run(find_counter=counter)
    _job['num_gpus'] = 0
    _job['num_cpus'] = 32
    _job['pgm_thread'] = 28
    _job['gb'] = 64
    _job['time'] = 4 # what time should this be?
    _job['tem_results_dir'] = os.path.join(tem_results_dir, tem_results_subdir)
    _job['pgm_proposals_dir'] = pgm_proposals_dir
    _job['pgm_features_dir'] = pgm_feats_dir
    
    _job['checkpoint_path'] = os.path.join(pgm_proposals_dir, tem_results_subdir)
    if not os.path.exists(_job['checkpoint_path']):
        os.makedirs(_job['checkpoint_path'], exist_ok=True)
        
    _job['pgm_score_threshold'] = 0.25
    _job['module'] = 'PGM'
    _job['mode'] = 'pgm'

    for key in ['checkpoint_path']:
        if key in _job:
            del _job[key]
            
    print(counter, sorted(_job.items()))
    fb_run_batch(_job, counter, email, code_directory)
