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
num_gpus = 1 # NOTE


def run(find_counter=None):
    counter = 950 # NOTE: adjust each time 451, 715, 750, 782, 814, 854
    
    for tem_results_subdir in sorted(os.listdir(tem_results_dir)):
        # if counter - start_counter > 100:
        #     print('Stopping at %d' % counter)
        #     break

        print(tem_results_subdir)
    
        _counter = int(regex.match(tem_results_subdir).groups()[0])

        job = temrun(find_counter=_counter)
        if type(job) == tuple:
            job = job[1]
        for key in list(job.keys()):
            if key.startswith('tem'):
                del job[key]
                
        name = job['name']
        for ckpt_subdir in os.listdir(os.path.join(tem_results_dir, tem_results_subdir)):
            _job = deepcopy(job)
            if 'thumos' in _job['dataset']:
                _job['video_anno'] = os.path.join(_job['video_info'], 'thumos_anno_action.json')
            elif 'gymnastics' in _job['dataset']:
                _job['video_anno'] = os.path.join(_job['video_info'], 'gymnastics_anno_action.sep052019.json')
            elif 'activitynet' in _job['dataset']:
                _job['video_anno'] = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/activitynet_annotations/anet_anno_action.json'
                
            _job['pgm_proposals_dir'] = os.path.join(pgm_proposals_dir, tem_results_subdir, ckpt_subdir)
            _job['pgm_features_dir'] = os.path.join(pgm_feats_dir, tem_results_subdir, ckpt_subdir)
            _job['module'] = 'PEM'
            _job['mode'] = 'train'
            _job['pem_compute_loss_interval'] = 1
            _job['pem_epoch'] = 40
            _job['pem_do_index'] = True
            if _job['dataset'] != 'activitynet':
                _job['video_info'] = os.path.join(_job['video_info'], 'Full_Annotation.csv')
            _job['checkpoint_path'] = checkpoint_path

            subname = tem_results_subdir.split(str(_counter))[1]
            subname = '%05d%s' % (_counter, subname)
            _job['name'] = '%s.%s' % (subname, ckpt_subdir)
            _job['num_gpus'] = num_gpus
            _job['num_cpus'] = num_gpus * 10
            _job['gb'] = 64 * num_gpus
            _job['time'] = 6
            _job['pem_feat_dim'] = 48
            _job['pem_batch_size'] = int(400 / num_gpus)

            for pem_training_lr in [0.01]:
                for pem_weight_decay in [0.0, 1e-4]:
                    for pem_l2_loss in [0.0, 0.000025]:
                        if pem_weight_decay > 0 and pem_l2_loss > 0:
                            continue
                        if pem_weight_decay == pem_l2_loss == 0.0:
                            continue
                        
                        for milestones in ['10,30', '10,20']:
                            for pem_step_gamma in [0.1, 0.5]:
                                counter += 1
                                
                                __job = deepcopy(_job)
                                __job['pem_training_lr'] = pem_training_lr
                                __job['pem_weight_decay'] = pem_weight_decay
                                __job['pem_l2_loss'] = pem_l2_loss
                                __job['pem_lr_milestones'] = milestones
                                __job['pem_step_gamma'] = pem_step_gamma
                                __job['name'] = '%s-%05d' % (_job['name'], counter)
                                
                                if not find_counter:
                                    func(__job, counter, email, code_directory)
                                elif counter == find_counter:
                                    return __job
    print(counter) # ended w 782, 814, 854, 950, 1054
    
if __name__ == '__main__':
    run()

    
