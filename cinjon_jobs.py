"""Run the jobs in this file.

Example commands:
python cinjon_jobs.fb cims batch: Run the func jobs on fb using sbatch.
"""
import os
import sys

from run_on_cluster import fb_run_batch

email = 'cinjon@nyu.edu'
code_directory = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch'
anno_directory = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/gymnastics_annotations'

func = fb_run_batch


def run(find_counter=None):
    counter = 0
    job = {
        'name': '2019.8.28.bsn',
        'video_anno': os.path.join(anno_directory, 'anno_fps12.on.json'),
        'video_info': os.path.join(anno_directory, 'video_info_new.csv'),
        'do_representation': True,
        'module': 'TEM',
        'time': 16,
        'tem_compute_loss_interval': 50
    }
    for tem_train_subset in ['overfit', 'train']:
        for num_gpus in [8, 4]:
            for lr in [1e-3, 3e-3, 1e-2]:
                for nvf in [25]:
                    for svf in [6, 8, 10]:
                        for num_run in range(2):
                            if num_gpus == 4 and num_run > 0:
                                continue
                    
                            counter += 1
                            _job = {k: v for k, v in job.items()}
                            _job['num_gpus'] = num_gpus
                            _job['tem_training_lr'] = lr
                            _job['name'] = '%s-%05d.%d' % (_job['name'], counter, num_run)
                            _job['num_cpus'] = num_gpus * 10
                            _job['gb'] = 64 * num_gpus
                            _job['num_videoframes'] = nvf
                            _job['skip_videoframes'] = svf
                            _job['tem_train_subset'] = tem_train_subset
                            _job['time'] = 1.1 if tem_train_subset == 'overfit' else 4
                            if find_counter == counter:
                                return _job
                            # func(_job, counter, email, code_directory)


    job = {
        'name': '2019.8.29.bsn',
        'video_anno': os.path.join(anno_directory, 'anno_fps12.on.json'),
        'video_info': os.path.join(anno_directory, 'video_info_new.csv'),
        'do_representation': True,
        'module': 'TEM',
        'tem_compute_loss_interval': 50
    }
    for tem_train_subset in ['overfit', 'train']:
        for num_gpus in [8]:
            for lr in [2e-3]:
                for nvf in [25, 30]:
                    for svf in [6, 8, 10]:
                        counter += 1
                        _job = {k: v for k, v in job.items()}
                        _job['num_gpus'] = num_gpus
                        _job['tem_training_lr'] = lr
                        _job['name'] = '%s-%05d' % (_job['name'], counter)
                        _job['num_cpus'] = num_gpus * 10
                        _job['gb'] = 64 * num_gpus
                        _job['num_videoframes'] = nvf
                        _job['skip_videoframes'] = svf
                        _job['tem_train_subset'] = tem_train_subset
                        _job['time'] = 3 if tem_train_subset == 'overfit' else 8
                        if find_counter == counter:
                            return _job
                        # func(_job, counter, email, code_directory)
                        


    job = {
        'name': '2019.8.30',
        'video_anno': os.path.join(anno_directory, 'anno_fps12.on.json'),
        'video_info': os.path.join(anno_directory, 'video_info_new.csv'),
        'do_representation': True,
        'module': 'TEM',
        'tem_compute_loss_interval': 50
    }
    for num_run in range(2):
        for tem_train_subset in ['overfit', 'train']:
            for num_gpus in [8]:
                for lr in [2e-3, 1e-2]:
                    for nvf in [25]:
                        for svf in [5]:
                            counter += 1
                            _job = {k: v for k, v in job.items()}
                            _job['num_gpus'] = num_gpus
                            _job['tem_training_lr'] = lr
                            _job['name'] = '%s-%05d.%d' % (_job['name'], counter, num_run)
                            _job['num_cpus'] = num_gpus * 10
                            _job['gb'] = 64 * num_gpus
                            _job['num_videoframes'] = nvf
                            _job['skip_videoframes'] = svf
                            _job['tem_train_subset'] = tem_train_subset
                            _job['time'] = 2
                            if find_counter == counter:
                                return _job
                            
                            if not find_counter:
                                func(_job, counter, email, code_directory)


if __name__ == '__main__':
    run()
