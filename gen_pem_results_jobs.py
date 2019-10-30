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

regex = re.compile('.*(\d{5}).ckpt.*-(\d{5}).*')
num_gpus = 4

matches = {
    775: 19, 525: 2, 447: 6, 435: 11, 507: 4, 483: 5, 567: 4,
    713: 3, 531: 0, 459: 3,
    108: 19, 66: 2, 3: 6, 10: 11, 130: 4, 35: 5, 147: 4,
    114: 3, 81: 0, 25: 3,
    828: 13, 906: 6, 898: 15, 872: 25, 836: 15, 856: 25,
    822: 37, 861: 14, 835: 3, 847: 15,
    943: 17, 950: 17, 976: 6, 977: 2, 928: 8, 1016: 6,
    1040: 8, 1000: 1, 1005: 1,
    1063: 35, 1051: 34, 1117: 10, 1045: 28, 1081: 35, 1094: 27, 1123: 13, 1106: 12, 1115: 16, 1095: 7, 1128: 20,
    1319: 31, 1201: 6, 1186: 7, 1286: 8, 1188: 5, 1228: 16
}


for ckpt_subdir in os.listdir(ckpt_directory):
    c1, c2 = regex.match(ckpt_subdir).groups()
    c1 = int(c1)
    c2 = int(c2)
    counter = c2
    _job = run(find_counter=counter)
    _job['num_gpus'] = num_gpus
    _job['num_cpus'] = num_gpus * 10
    _job['gb'] = 64 * num_gpus
    _job['time'] = 3
    _job['pem_inference_results_dir'] = pem_results_dir
    _job['pem_inference_subset'] = 'full'
    _job['mode'] = 'inference'
    _job['checkpoint_path'] = os.path.join(ckpt_directory, ckpt_subdir)

    name = _job['name']
    ckpt_epoch = matches[c1]
    _job['checkpoint_epoch'] = ckpt_epoch
    _job['name'] = '%s.ckpt%d' % (name, ckpt_epoch)
    print(ckpt_subdir, counter, _job['name'])
    # print(sorted(_job.items()))
    fb_run_batch(_job, counter, email, code_directory)
