"""Run the jobs in this file.

Example commands:
python pem_jobs.py
"""
from copy import deepcopy
import os
import re
import sys

from run_on_cluster import fb_run_batch
from tem_jobs import run as temrun

email = 'cinjon@nyu.edu'
code_directory = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch'
gymnastics_anno_directory = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/gymnastics_annotations'
thumos_anno_directory = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/thumos14_annotations'

base_dir = '/checkpoint/cinjon/spaceofmotion/bsn'
checkpoint_path = os.path.join(base_dir, 'checkpoint', 'pem')
tem_dir = os.path.join(base_dir, 'teminf')
tem_results_dir = os.path.join(tem_dir, 'results')
pgm_proposals_dir = os.path.join(base_dir, 'pgmprops')
pgm_feats_dir = os.path.join(base_dir, 'pgmfeats')

regex = re.compile('.*(\d{5}).*')

func = fb_run_batch
num_gpus = 4 # NOTE


def run(find_counter=None):
    counter = 0

    for tem_results_subdir in os.listdir(tem_results_dir):
        _counter = int(regex.match(tem_results_subdir).groups()[0])
        job = temrun(find_counter=_counter)
        for key in list(job.keys()):
            if key.startswith('tem'):
                del job[key]
                
        name = job['name']
        for ckpt_subdir in os.listdir(os.path.join(tem_results_dir, tem_results_subdir)):
            _job = deepcopy(job)
            if 'thumos' in _job['dataset']:
                _job['video_anno'] = os.path.join(_job['video_info'], 'thumos_anno_action.json')

            _job['pgm_proposals_dir'] = os.path.join(pgm_proposals_dir, tem_results_subdir, ckpt_subdir)
            _job['pgm_features_dir'] = os.path.join(pgm_feats_dir, tem_results_subdir, ckpt_subdir)
            _job['module'] = 'PEM'
            _job['mode'] = 'train'
            _job['pem_compute_loss_interval'] = 5
            _job['pem_epoch'] = 40
            _job['pem_do_index'] = True
            if 'thumos' in _job['dataset']:
                _job['video_info'] = os.path.join(_job['video_info'], 'Full_Annotation.csv')
            _job['checkpoint_path'] = checkpoint_path
            
            _job['name'] = '2019.09.17.%s.%s' % (tem_results_subdir, ckpt_subdir)
            _job['num_gpus'] = num_gpus
            _job['num_cpus'] = num_gpus * 10
            _job['gb'] = 64 * num_gpus
            _job['time'] = 1
            _job['pem_feat_dim'] = 48
            
            for pem_batch_size in [64]:
                for pem_training_lr in [1e-3, 3e-3]:
                    for pem_weight_decay in [1e-4]: #, 3e-4]:
                        for pem_step_gamma in [0.5]:
                            for pem_step_size in [10]:
                                for pem_hidden_dim in [256]:
                                    for pem_max_zero_weight in [0.25, 0.1]:
                                        for pem_top_K in [1500, -1]:
                                            counter += 1
                                            __job = deepcopy(_job)
                                            __job['pem_training_lr'] = pem_training_lr
                                            __job['pem_batch_size'] = pem_batch_size
                                            __job['pem_step_size'] = pem_step_size
                                            __job['pem_step_gamma'] = pem_step_gamma
                                            __job['pem_weight_decay'] = pem_weight_decay
                                            __job['pem_hidden_dim'] = pem_hidden_dim
                                            __job['pem_max_zero_weight'] = pem_max_zero_weight
                                            __job['pem_top_K'] = pem_top_K
                                            __job['pem_top_K_inference'] = pem_top_K
                                            __job['name'] = '%s-%05d' % (_job['name'], counter)

                                            if not find_counter:
                                                if pem_top_K > 0:
                                                    continue
                                                func(__job, counter, email, code_directory)
                                            elif counter == find_counter:
                                                return __job
    print(counter)
    
if __name__ == '__main__':
    run()

    
