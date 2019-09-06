"""Run the jobs in this file.

Example commands:
python pem_jobs.py
"""
import os
import sys

from run_on_cluster import fb_run_batch

email = 'cinjon@nyu.edu'
code_directory = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch'
anno_directory = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/gymnastics_annotations'

func = fb_run_batch
num_gpus = 8


def run(find_counter=None):
    counter = 0

    job = {
        'name': '2019.09.05',
        'video_anno': os.path.join(anno_directory, 'anno_fps12.on.json'),
        'video_info': os.path.join(anno_directory, 'video_info_new.csv'),
        'module': 'PEM',
        'mode': 'train',
        'pem_compute_loss_interval': 100,
        'pgm_proposals_dir': '/checkpoint/cinjon/spaceofmotion/bsn/pgmprops/101.2019.8.30-00101.1',
        'pgm_features_dir': '/checkpoint/cinjon/spaceofmotion/bsn/pgmfeats/101.2019.8.30-00101.1',
        'pem_epoch': 40,
    }
    for pem_batch_size in [1024, 2048]:
        for pem_training_lr in [3e-3, 1e-2, 3e-2]:
            for pem_weight_decay in [1e-5, 3e-4, 1e-4]:
                for pem_step_gamma in [0.5, 0.75]:
                    for pem_step_size in [5, 7, 10]:
                        for pem_hidden_dim in [256, 1024]:
                            for pem_top_threshold in [0.95, 0.9, 0.75]:
                                counter += 1
                                
                                _job = {k: v for k, v in job.items()}
                                _job['num_gpus'] = num_gpus
                                _job['pem_training_lr'] = pem_training_lr
                                _job['pem_batch_size'] = pem_batch_size
                                _job['pem_step_size'] = pem_step_size
                                _job['pem_step_gamma'] = pem_step_gamma
                                _job['pem_weight_decay'] = pem_weight_decay
                                _job['pem_hidden_dim'] = pem_hidden_dim
                                _job['pem_top_threshold'] = pem_top_threshold
                                _job['pem_feat_dim'] = 48
                            
                                _job['name'] = '%s-%05d' % (_job['name'], counter)
                                _job['num_cpus'] = num_gpus * 10
                                _job['gb'] = 64 * num_gpus
                                _job['time'] = 4

                                # if not find_counter:
                                #     func(_job, counter, email, code_directory)


    job = {
        'name': '2019.09.06',
        'video_anno': os.path.join(anno_directory, 'anno_fps12.on.json'),
        'video_info': os.path.join(anno_directory, 'video_info_new.csv'),
        'module': 'PEM',
        'mode': 'train',
        'pem_compute_loss_interval': 5,
        'pgm_proposals_dir': '/checkpoint/cinjon/spaceofmotion/bsn/pgmprops/101.2019.8.30-00101.1',
        'pgm_features_dir': '/checkpoint/cinjon/spaceofmotion/bsn/pgmfeats/101.2019.8.30-00101.1',
        'pem_epoch': 60,
        'pem_do_index': True,
    }
    for pem_batch_size in [64, 128, 512]:
        for pem_training_lr in [3e-3, 1e-2, 3e-2]:
            for pem_weight_decay in [1e-5, 1e-4]:
                for pem_step_gamma in [0.5]:
                    for pem_step_size in [10]:
                        for pem_hidden_dim in [256, 1024, 2048]:
                            for pem_max_zero_weight in [0.25, 0.1]:
                                for pem_top_K in [1250, 2500]:
                                    counter += 1
                                
                                    _job = {k: v for k, v in job.items()}
                                    _job['num_gpus'] = num_gpus
                                    _job['pem_training_lr'] = pem_training_lr
                                    _job['pem_batch_size'] = pem_batch_size
                                    _job['pem_step_size'] = pem_step_size
                                    _job['pem_step_gamma'] = pem_step_gamma
                                    _job['pem_weight_decay'] = pem_weight_decay
                                    _job['pem_hidden_dim'] = pem_hidden_dim
                                    _job['pem_max_zero_weight'] = pem_max_zero_weight
                                    _job['pem_top_K'] = pem_top_K
                                    _job['pem_top_K_inference'] = pem_top_K
                                    _job['pem_feat_dim'] = 48
                                    
                                    _job['name'] = '%s-%05d' % (_job['name'], counter)
                                    _job['num_cpus'] = num_gpus * 10
                                    _job['gb'] = 64 * num_gpus
                                    _job['time'] = 3
                                    
                                    if not find_counter:
                                        func(_job, counter, email, code_directory)

    
if __name__ == '__main__':
    run()

    
