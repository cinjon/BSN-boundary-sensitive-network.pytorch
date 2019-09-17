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
# anno_directory = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/gymnastics_annotations'
base_dir = '/checkpoint/cinjon/spaceofmotion/bsn'
tem_dir = os.path.join(base_dir, 'teminf')
tem_results_dir = os.path.join(tem_dir, 'results')
if not os.path.exists(tem_results_dir):
    os.makedirs(tem_results_dir)
ckpt_directory = os.path.join(tem_dir, 'do_ckpts')

regex = re.compile('.*(\d{5}).*')

now = datetime.datetime.now()
print(now)

for ckpt_subdir in os.listdir(ckpt_directory):
    counter = int(regex.match(ckpt_subdir).groups()[0])
    if counter not in [195]:
        continue

    _job = run(find_counter=counter)
    _job['num_gpus'] = 8
    _job['num_cpus'] = 8 * 10
    _job['gb'] = 64 * 8
    _job['time'] = 3
    _job['tem_results_dir'] = tem_results_dir
    _job['mode'] = 'inference'
    
    _job['checkpoint_path'] = os.path.join(ckpt_directory, ckpt_subdir)
    _job['tem_results_subset'] = 'full'
    name = _job['name']
    for ckpt_epoch in [5, 15, 20]:
        if counter == 195:
            if ckpt_epoch < 20:
                continue
            ckpt_epoch = 19
            
        _job['checkpoint_epoch'] = ckpt_epoch
        _job['name'] = '%s.ckpt%d' % (name, ckpt_epoch)
        print(ckpt_subdir, counter)
        print(sorted(_job.items()))
        fb_run_batch(_job, counter, email, code_directory)
    print('\n')        

print(now)
