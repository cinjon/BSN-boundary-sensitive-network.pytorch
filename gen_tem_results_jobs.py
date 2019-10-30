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
    447: 4, 725: 4, 775: 6, 713: 6, 737: 4,

    861: 5, 872: 5, 856: 5, 828: 7, 847: 8, 836: 5, 822: 21, 835: 5,
    906: 5, 898: 1,
    950: 2, 943: 2, 928: 1, 976: 15, 1000: 14, 1005: 12, 1016: 9,
    977: 3, 1040: 15,
    1128: 3, 1115: 2, 1106: 8, 1094: 8, 1117: 28, 1123: 2, 1095: 2, 1051: 1, 1045: 1, 1063: 1, 1081: 1,
    1180: 1, 1156: 1, 1147: 2, 1188: 3, 1186: 3, 1201: 4, 1228: 3,
    1286: 2, 1319: 1, 1252: 13
}


num_gpus = 4
for ns, ckpt_subdir in enumerate(sorted(os.listdir(ckpt_directory))):
    counter = int(regex.match(ckpt_subdir).groups()[0])
    print(counter, ckpt_subdir)
    _job = run(find_counter=counter)
    _job['num_gpus'] = num_gpus
    _job['num_cpus'] = num_gpus * 6 # 10
    _job['gb'] = 64 * num_gpus
    _job['time'] = 12
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
print(ns)
