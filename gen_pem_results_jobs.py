"""Run the jobs that generate TEM Results.

Example commands:
python gen_tem_results_jobs.py
"""
import json
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
    1319: 31, 1201: 6, 1186: 7, 1286: 8, 1188: 5, 1228: 16,
    2055: 24, 2057: 15, 2093: 16, 1937: 16, 1944: 4, 1949: 9, 1950: 12, 1935: 11, 1994: 19, 1971: 20, 2037: 20, 2031: 26, 2179: 13, 2189: 26, 2186: 16, 2145: 16,
    # TSN Gymnastics
    3720: 9, 3684: 20, 3726: 2, 3705: 25, 3687: 14,
    # Corrflow NFC anet
    3471: 29, 3477: 16, 3465: 23, 3468: 14, 3483: 13,
    # ResNet NFC anet
    3357: 15, 3387: 5, 3381: 7, 3351: 12, 3369: 27,
    # Resnet dfc anet
    3405: 10, 3423: 16, 3435: 4, 3393: 2, 3429: 12, 3417: 23,
    # CCC anet
    3543: 13, 3561: 24, 3567: 6, 3549: 8, 3573: 17,
    # Ugh I'm silly. Here's the CorrfloW DFC anet that I never put in
    3513: 13, 3489: 19, 3525: 11, 3501: 9, 3531: 11, 3495: 26, 3507: 17,
    # AMDIM DFC anet
    3639: 21, 3651: 2, 3633: 4, 3645: 3,
    # TSN RGB 1
    3975: 28, 3984: 6,
    # TSN RGB 2
    4005: 1, 4006: 15, 4003: 17, 4002: 6, 4001: 9,
    # TSN RGB 3
    4593: 21, 4611: 14, 4599: 14,  # 4596 got lost in the shuffle :(
    4659: 30, 4656: 23, 4647: 36, 4650: 22,
    4623: 30, 4620: 23, 4632: 27, 4617: 28,
    # CorrFLow NFC NF:
    4802: 1, 4779: 16,
    # CCC FT
    4560: 34, 4584: 32, 4572: 4, 4575: 3, 4554: 36, 4578: 14
}


fixed = json.load(open('/checkpoint/cinjon/spaceofmotion/bsn/peminf/fixeddata.json', 'r'))
# print('Fixed: ', len(fixed), fixed.keys())
# print(sorted(fixed.keys()))
# fixed = {int(k.split('.')[0]): {i:j for i, j in v.items() if not i.startswith('base') and not 'curr_' in i and not 'start_time' in i} for k, v in fixed.items()}
# print(sorted(fixed.items())[0])


check = 0
goods = []
bads = []
for ckpt_subdir in os.listdir(ckpt_directory):
    c1, c2 = regex.match(ckpt_subdir).groups()    
    c1 = int(c1)
    c2 = int(c2)
    counter = c2
    # print('\n', ckpt_subdir, '\n')
    if c1 in fixed:
        print('Got from fixed')
        _job = fixed[c1]
    elif str(c1) in fixed:
        print('Got from fixed str')
        _job = fixed[str(c1)]
    else:
        print('Run...')
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
    check += 1
    # print(sorted(_job.items()), '\n')
    fb_run_batch(_job, counter, email, code_directory)
print(check)
