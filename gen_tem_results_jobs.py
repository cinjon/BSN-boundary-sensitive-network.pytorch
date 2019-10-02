"""Run the jobs that generate TEM Results.

Example commands:
python gen_tem_results_jobs.py
"""
import datetime
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
if not os.path.exists(tem_results_dir):
    os.makedirs(tem_results_dir)
ckpt_directory = os.path.join(tem_dir, 'do_ckpts')

regex = re.compile('.*(\d{5}).*')

matches = {
    485: 5, 465: 3, 567: 6, 483: 1, 487: 6, 473: 3, 461: 5, 557: 5,
    559: 1, 525: 3, 519: 2, 435: 4, 459: 1, 507: 3, 453: 1, 531: 4,
    447: 4, 725: 4, 775: 6, 713: 6, 737: 4
}

num_gpus = 4
for ckpt_subdir in sorted(os.listdir(ckpt_directory)):
    counter = int(regex.match(ckpt_subdir).groups()[0])
    if counter in [775, 713, 737]:
        continue

    _job = run(find_counter=counter)
    _job['num_gpus'] = num_gpus
    _job['num_cpus'] = num_gpus * 6 # 10
    _job['gb'] = 64 * num_gpus
    _job['time'] = 14
    _job['tem_results_dir'] = tem_results_dir
    _job['mode'] = 'inference'
    
    _job['checkpoint_path'] = os.path.join(ckpt_directory, ckpt_subdir)
    _job['tem_results_subset'] = 'full'
    name = _job['name']
    ckpt_epoch = matches[counter]
    _job['checkpoint_epoch'] = ckpt_epoch
    _job['name'] = '%s.ckpt%d' % (name, ckpt_epoch)
    print(ckpt_subdir, counter)
    if 'tem_batch_size' not in _job:
        _job['tem_batch_size'] = 4 if _job['do_feat_conversion'] else 1
    print(sorted(_job.items()))
    fb_run_batch(_job, counter, email, code_directory)
    print('\n')
