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
    1286: 2, 1319: 1, 1252: 13,
    # Below is redoign gymnastics w/o sampler.
    2093: 9, 2057: 8, 2055: 14, 1944: 8, 1935: 5, 1950: 7, 1949: 18, 1937: 18, 1971: 15, 1994: 17, 2031: 6, 2037: 6, 2189: 10, 2145: 4, 2179: 10, 2186: 10, 2097: 1, 2132: 1,
    # Below is TSN on Gymnastics.
    3705: 2, 3720: 4, 3726: 6, 3684: 4, 3687: 2,
    # Below is ResNet on ActivityNet:
    3393: 21, 3423: 18, 3405: 9, 3435: 18, 3417: 11, 3429: 8, 3357: 15, 3369: 9, 3381: 11, 3351: 25, 3387: 22,
    # Below is Corrflow on ActivityNet:
    3513: 12, 3531: 11, 3507: 21, 3525: 8, 3489: 14, 3501: 9, 3495: 16, 3483: 17, 3477: 13, 3471: 20, 3465: 19, 3468: 19 ,
    # Below is CCC on ActivityNet:
    3561: 18, 3543: 12, 3567: 18, 3549: 19, 3573: 11,
    # Below is AMDIM on ActivityNet:
    3645: 11, 3633: 7, 3639: 8, 3651: 8,
    # Below is CCC NFC
    3759: 23, 3786: 25,
    # TSN RGB Gym
    3975: 4, 3984: 4,
    # TSN RGB Gym 2
    4005: 4, 4001: 6, 4003: 6, 4006: 14, 4002: 4,
    # TSN RGB Gym 3 lulz. And a bunch more.
    4596: 5, 4593: 14, 4611: 7, 4599: 7, # DFC
    4659: 5, 4656: 1, 4647: 5, 4650: 1, # NFC Reg
    4623: 3, 4620: 3, 4632: 2, 4617: 3, # NFC NF
    # CCC FT DFC Gym:
    4575: 4, 4578: 11, 4584: 4, 4554: 10, 4560: 11, 4572: 14,
    # AMDIM NFC:
    4440: 1,
    # AMDIM NF:
    4695: 1,
    # Corrflow NF
    4793: 14, 4779: 2, 4776: 2, 4802: 14, 4792: 11,
    # TSN Thumos NFC Reg
    4863: 2, 4872: 1,
    # TSN Thumos NFC NF
    4908: 8, 4905: 8, 4911: 8, 4926: 8, 4923: 8, 4914: 8,
    # TSN Thumos DFC
    4881: 1, 4896: 3, 4899: 1, 4902: 3,
    # DFC Resnet Reg
    4971: 8, 4956: 3, 4965: 7, 4974: 6,
    # DFC Resnet-Rand
    5256: 19, 5244: 14, 5247: 13, 5238: 21, 5241: 22, 5262: 10, 5253: 7,
    # NFC Resnet-Rand Reg
    5193: 21, 5175: 12, 5187: 18, 5181: 13, 5211: 10, 5199: 8, # In the queue.
    # DFC Resnet NotRand:
    4989: 2, 4980: 3,
    # DFC Resnet Rand
    5343: 11, 5355: 25, 5319: 16, 5325: 24, 5349: 16,
    # NFC Reg ResNet Rand
    5307: 17, 5271: 17, 5274: 24, 5295: 13, 5277: 25, # done
    # TSN Gym Rand NFC:
    5370: 7, 5388: 5, 5373: 4, 5391: 7,
    # TSN Rand DFC 
    5442: 3, 5445: 3, # done
    # CorrFlow Rand DFC Gymnastics
    5712: 7, 5733: 5, 5742: 7, 5718: 19,
    # Corrflow Rand DFC ThumosImages
    5823: 10, 5802: 10, 5832: 13, 5808: 18,
    # CCC Thumos Rand dFC
    5646: 9, 5634: 21, 5619: 4, 5640: 9, 5628: 13, # done
    # Thumos Resnet Rand NL DFC:
    6108: 10, 6123: 8, 6102: 16, 6090: 7, 6081: 11, 6114: 14,
    # Thumos Resnet Reg NL DFC:
    6030: 3, 6027: 3, 6015: 3, 6018: 3,
    # Gymnastics Resnet Rand NL DFC:
    6054: 10, 6048: 8, 6075: 8, 6069: 18, 6057: 9, 6078: 9, 6039: 24, # done
    # TSN Rnad DFC Thumos: 
    5538: 3, 5547: 2,
    # Thumos Rand TSN NFC
    5475: 1, 5478: 1,
    # Gymnastics CCC Reg Rand:
    5565: 4, 5586: 4, 5559: 4, 5598: 4, 5571: 4, 5580: 4
}


num_gpus = 8
check = 0
for ns, ckpt_subdir in enumerate(sorted(os.listdir(ckpt_directory))):
    counter = int(regex.match(ckpt_subdir).groups()[0])
    
    print(counter, ckpt_subdir)

    _job = run(find_counter=counter)
    if type(_job) == tuple:
        _job = _job[1]
        
    _job['num_gpus'] = num_gpus
    _job['num_cpus'] = num_gpus * 6 # 10
    _job['gb'] = 64 * num_gpus
    _job['time'] = 5
    _job['tem_results_dir'] = tem_results_dir
    _job['mode'] = 'inference'
    
    _job['checkpoint_path'] = os.path.join(ckpt_directory, ckpt_subdir)
    _job['tem_results_subset'] = 'full'
    name = _job['name']
    ckpt_epoch = matches[counter]
    _job['checkpoint_epoch'] = ckpt_epoch
    print(ckpt_subdir, counter, _job['name'])
    _job['name'] = '%s.ckpt%d' % (name, ckpt_epoch)
    if 'tem_batch_size' not in _job:
        _job['tem_batch_size'] = 4 if _job['do_feat_conversion'] else 1
    if _job['dataset'] == 'activitynet':
        _job['time'] = 10
    print(_job['dataset'], _job['representation_module'], _job['do_feat_conversion'])
    # print(sorted(_job.items()))
    check += 1
    fb_run_batch(_job, counter, email, code_directory)
    print('\n')                
print(ns+1, check)
