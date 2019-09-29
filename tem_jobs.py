"""Run the jobs in this file.

Example running all:

python tem_jobs.py

Example for running thumosfeatures:

python main.py --module TEM --mode train --name dbg.thumos --counter 0 --data_workers 8 --seed 0 --num_gpus 2 --checkpoint_path /checkpoint/cinjon/spaceofmotion/bsn/checkpoint --video_info /private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/thumos14_annotations --feature_dirs /private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/thumos14_feature_anet_200/flow/csv,/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/thumos14_feature_anet_200/rgb/csv  --dataset thumosfeatures
"""
import os
import sys

from run_on_cluster import fb_run_batch

base_dir = '/checkpoint/cinjon/spaceofmotion/bsn'
checkpoint_path = os.path.join(base_dir, 'checkpoint', 'tem')
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
                            if counter in [69, 70, 73, 74]:
                                continue
                            
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
                            
                            # if not find_counter:
                            #     func(_job, counter, email, code_directory)


    # Not sure what happened in teh above. The below is redoing but letting training be over many more frames.
    # Pretty much changed training to do frames [0 + skip*nf, skip + skip*nf, 2*skip + skip*nf, ...]
    # but enforce that testing is only on windows [0 + skip*nf, skip*nf + skip*nf, 2*skip*nf + skip*nf, ...],
    # i.e. non-overlapping windows.
    job = {
        'name': '2019.8.30',
        'video_anno': os.path.join(anno_directory, 'anno_fps12.on.json'),
        'video_info': os.path.join(anno_directory, 'video_info_new.csv'),
        'do_representation': True,
        'module': 'TEM',
        'tem_compute_loss_interval': 50,
        'num_videoframes': 25,
        'skip_videoframes': 5,
    }
    for num_run in range(2):
        for tem_train_subset in ['overfit', 'train']:
            for num_gpus in [8]:
                for lr in [2e-3, 6e-3]:
                    for tem_step_gamma in [0.1, 0.5]:
                        for tem_step_size in [7, 10]:
                            counter += 1                            
                            _job = {k: v for k, v in job.items()}
                            _job['num_gpus'] = num_gpus
                            _job['tem_training_lr'] = lr
                            _job['name'] = '%s-%05d.%d' % (_job['name'], counter, num_run)
                            _job['num_cpus'] = num_gpus * 10
                            _job['gb'] = 64 * num_gpus
                            _job['tem_step_gamma'] = tem_step_gamma
                            _job['tem_step_size'] = tem_step_size
                            _job['tem_train_subset'] = tem_train_subset
                            if tem_train_subset == 'overfit':
                                time = 2
                            elif tem_train_subset == 'train' and num_run == 0:
                                time = 3
                            else:
                                time = 13
                            _job['time'] = time
                            
                            if find_counter == counter:
                                return _job
                            
                            # if not find_counter:
                            #     func(_job, counter, email, code_directory)


    job = {
        'name': '2019.09.02',
        'video_anno': os.path.join(anno_directory, 'anno_fps12.on.json'),
        'video_info': os.path.join(anno_directory, 'video_info_new.csv'),
        'do_representation': True,
        'module': 'TEM',
        'tem_compute_loss_interval': 50,
        'num_videoframes': 25,
        'skip_videoframes': 5,
        'tem_step_size': 10,
        'tem_step_gamma': 0.5
    }
    for num_run in range(2):
        for tem_train_subset in ['overfit', 'train']:
            for num_gpus in [8]:
                for lr in [2e-3, 4e-3, 6e-3, 1e-2]:
                    if tem_train_subset == 'overfit' and num_run > 0:
                        continue
                    if tem_train_subset == 'overfit' and lr < 6e-3:
                        continue
                    if tem_train_subset == 'train' and lr > 4e-3:
                        continue
                
                    counter += 1                            
                    _job = {k: v for k, v in job.items()}
                    _job['num_gpus'] = num_gpus
                    _job['tem_training_lr'] = lr
                    _job['name'] = '%s-%05d.%d' % (_job['name'], counter, num_run)
                    _job['num_cpus'] = num_gpus * 10
                    _job['gb'] = 64 * num_gpus
                    _job['tem_train_subset'] = tem_train_subset
                    if tem_train_subset == 'overfit':
                        time = 8
                    elif num_run == 0:
                        time = 8
                    else:
                        time = 18
                    _job['time'] = time
                    
                    if find_counter == counter:
                        return _job
                    
                    if not find_counter:
                        pass
                        # func(_job, counter, email, code_directory)


    # print("Counter: ", counter)
    job = {
        'name': '2019.09.06',
        'video_info': '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/thumos14_annotations',
        'feature_dirs': '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/thumos14_feature_anet_200/flow/csv,/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/thumos14_feature_anet_200/rgb/csv',
        'dataset': 'thumosfeatures',
        'module': 'TEM',
        'mode': 'train',
        'tem_compute_loss_interval': 10,
        'tem_step_size': 10,
        'tem_step_gamma': 0.5
    }
    for num_runs in range(2):
        for num_gpus in [8]:
            for lr in [1e-3, 3e-3, 1e-2]:
                for tem_step_gamma in [0.1, 0.5]:
                    for tem_step_size in [7, 10]:
                        for tem_weight_decay in [1e-4, 3e-4, 1e-5]:
                            counter += 1                            
                            _job = {k: v for k, v in job.items()}
                            _job['num_gpus'] = num_gpus
                            _job['tem_training_lr'] = lr
                            _job['tem_step_gamma'] = tem_step_gamma
                            _job['tem_weight_decay'] = tem_weight_decay
                            _job['tem_step_size'] = tem_step_size
                            _job['name'] = '%s-%05d' % (_job['name'], counter)
                            _job['num_cpus'] = num_gpus * 10
                            _job['gb'] = 64 * num_gpus
                            _job['time'] = 8 if num_runs == 0 else 16
                            
                            if find_counter == counter:
                                return _job
                    
                            # if not find_counter:
                            #     func(_job, counter, email, code_directory)


    # print("Counter: ", counter) # 184.
    # The jobs below are trying to dupliacte the TEM settings from the paper.
    job = {
        'name': '2019.09.10',
        'video_info': '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/thumos14_annotations',
        'feature_dirs': '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/thumos14_feature_anet_200/flow/csv,/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/thumos14_feature_anet_200/rgb/csv',
        'dataset': 'thumosfeatures',
        'module': 'TEM',
        'mode': 'train',
        'tem_compute_loss_interval': 10,
        'tem_training_lr': 1e-3,
        'tem_step_size': 10,
        'tem_step_gamma': 0.1,
        'tem_epoch': 20,
        'tem_batch_size': 16,
    }
    for num_run in range(2):
        for num_gpus in [1, 4]:
            for weight_decay in [1e-4, 5e-4, 1e-3, 5e-3]:
                counter += 1                            
                _job = {k: v for k, v in job.items()}
                _job['num_gpus'] = num_gpus
                _job['tem_weight_decay'] = tem_weight_decay
                _job['name'] = '%s-%05d' % (_job['name'], counter)
                _job['num_cpus'] = num_gpus * 10
                _job['gb'] = 64 * num_gpus
                _job['time'] = 3
                            
                if find_counter == counter:
                    return _job
                # elif not find_counter:
                #     func(_job, counter, email, code_directory)


    # print("Counter: ", counter)  # 200
    # The jobs below are trying to do ThumosImages w CorrFlow.
    # They use the representation change.
    job = {
        'name': '2019.09.10',
        'video_info': '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/thumos14_annotations',
        'dataset': 'thumosimages',
        'module': 'TEM',
        'mode': 'train',
        'tem_compute_loss_interval': 10,
        'tem_step_size': 10,
        'tem_epoch': 21,
        'tem_batch_size': 8,
        'do_representation': True,
        'do_feat_conversion': True,
        'representation_module': 'corrflow',
        'representation_checkpoint': '/checkpoint/cinjon/spaceofmotion/supercons/corrflow.kineticsmodel.pth',
    }
    for num_run in range(2):
        for num_gpus in [8]:
            for weight_decay in [1e-4, 5e-4, 1e-3, 5e-3]:
                for tem_training_lr in [1e-3, 3e-3]:
                    for tem_step_gamma in [0.75, 0.5, 0.1]:
                        counter += 1
                        _job = {k: v for k, v in job.items()}
                        _job['num_gpus'] = num_gpus
                        _job['tem_weight_decay'] = tem_weight_decay
                        _job['tem_training_lr'] = tem_training_lr
                        _job['tem_step_gamma'] = tem_step_gamma
                        _job['name'] = '%s-%05d' % (_job['name'], counter)
                        _job['num_cpus'] = num_gpus * 10
                        _job['gb'] = 64 * num_gpus
                        _job['time'] = 5 if num_run == 0 else 15
                            
                        if find_counter == counter:
                            return _job
                        # elif not find_counter:
                        #     func(_job, counter, email, code_directory)


    # print("Counter: ", counter)   # 248
    # The jobs below are trying to do ThumosImages w CorrFlow.
    # They however use the FULL representation from CF, which has
    # a size of 225280. Need to reduce the batch size to match.
    job = {
        'name': '2019.09.10',
        'video_info': '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/thumos14_annotations',
        'dataset': 'thumosimages',
        'module': 'TEM',
        'mode': 'train',
        'tem_compute_loss_interval': 10,
        'tem_step_size': 10,
        'tem_epoch': 21,
        'tem_batch_size': 4,
        'do_representation': True,
        'do_feat_conversion': False,
        'representation_module': 'corrflow',
        'representation_checkpoint': '/checkpoint/cinjon/spaceofmotion/supercons/corrflow.kineticsmodel.pth',
    }
    for num_run in range(2):
        for num_gpus in [8]:
            for weight_decay in [1e-4, 5e-4, 1e-3, 5e-3]:
                for tem_training_lr in [1e-3, 3e-3]:
                    for tem_step_gamma in [0.75, 0.5, 0.1]:
                        counter += 1                            
                        _job = {k: v for k, v in job.items()}
                        _job['num_gpus'] = num_gpus
                        # Sigh, fuck me.
                        _job['tem_weight_decay'] = tem_weight_decay
                        _job['tem_training_lr'] = tem_training_lr
                        _job['tem_step_gamma'] = tem_step_gamma
                        _job['name'] = '%s-%05d' % (_job['name'], counter)
                        _job['num_cpus'] = num_gpus * 10
                        _job['gb'] = 64 * num_gpus
                        _job['time'] = 5 if num_run == 0 else 15
                            
                        if find_counter == counter:
                            return _job
                        # elif not find_counter:
                        #     func(_job, counter, email, code_directory)

    # print("Counter: ", counter)   # 296
    # The jobs below are trying to do ThumosImages w CorrFlow.
    # They use the representation change.
    job = {
        'name': '2019.09.10',
        'video_info': '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/thumos14_annotations',
        'dataset': 'thumosimages',
        'module': 'TEM',
        'mode': 'train',
        'tem_compute_loss_interval': 10,
        'tem_step_size': 10,
        'tem_epoch': 21,
        'tem_batch_size': 8,
        'do_representation': True,
        'do_feat_conversion': True,
        'representation_module': 'corrflow',
        'representation_checkpoint': '/checkpoint/cinjon/spaceofmotion/supercons/corrflow.kineticsmodel.pth',
    }
    for num_run in range(2):
        for num_gpus in [8]:
            for tem_weight_decay in [1e-4, 5e-4, 1e-3]:
                for tem_training_lr in [1e-3, 3e-4]:
                    for tem_step_gamma in [0.75, 0.5, 0.1]:
                        counter += 1
                        _job = {k: v for k, v in job.items()}
                        _job['num_gpus'] = num_gpus
                        _job['tem_weight_decay'] = tem_weight_decay
                        _job['tem_training_lr'] = tem_training_lr
                        _job['tem_step_gamma'] = tem_step_gamma
                        _job['name'] = '%s-%05d' % (_job['name'], counter)
                        _job['num_cpus'] = num_gpus * 10
                        _job['gb'] = 64 * num_gpus
                        _job['time'] = 5
                            
                        if find_counter == counter:
                            return _job
                        # elif not find_counter:
                        #     func(_job, counter, email, code_directory)
                        

    # print("Counter: ", counter)  # 332
    # The jobs below are trying to do ThumosImages w CorrFlow.
    # They however use the FULL representation from CF, which has
    # a size of 225280. Need to reduce the batch size to match.
    job = {
        'name': '2019.09.10',
        'video_info': '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/thumos14_annotations',
        'dataset': 'thumosimages',
        'module': 'TEM',
        'mode': 'train',
        'tem_compute_loss_interval': 10,
        'tem_step_size': 10,
        'tem_epoch': 40,
        'tem_batch_size': 4,
        'do_representation': True,
        'do_feat_conversion': False,
        'representation_module': 'corrflow',
        'representation_checkpoint': '/checkpoint/cinjon/spaceofmotion/supercons/corrflow.kineticsmodel.pth',
    }
    for num_run in range(2):
        for num_gpus in [8]:
            for tem_weight_decay in [1e-4, 5e-4, 1e-3]:
                for tem_training_lr in [1e-3, 3e-4]:
                    for tem_step_gamma in [0.75, 0.5, 0.1]:
                        counter += 1                            
                        _job = {k: v for k, v in job.items()}
                        _job['num_gpus'] = num_gpus
                        _job['tem_weight_decay'] = tem_weight_decay
                        _job['tem_training_lr'] = tem_training_lr
                        _job['tem_step_gamma'] = tem_step_gamma
                        _job['name'] = '%s-%05d' % (_job['name'], counter)
                        _job['num_cpus'] = num_gpus * 10
                        _job['gb'] = 64 * num_gpus
                        _job['time'] = 8
                            
                        if find_counter == counter:
                            return _job
                        # elif not find_counter:
                        #     func(_job, counter, email, code_directory)


    # print("Counter: ", counter)  # 368
    # These are doing the full gymnastics dataset.
    job = {
        'name': '2019.09.15',
        'video_anno': os.path.join(anno_directory, 'anno_fps12.on.sep052019.json'),
        'video_info': os.path.join(anno_directory, 'video_info.sep052019.fps12.csv'),
        'do_representation': True,
        'module': 'TEM',
        'mode': 'train',
        'tem_compute_loss_interval': 10,
        'num_videoframes': 100,
        'skip_videoframes': 2,
        'tem_batch_size': 4,
        'tem_step_size': 10,
        'tem_step_gamma': 0.5,
        'tem_epoch': 30,
        'tem_train_subset': 'train',
        'do_representation': True,
        'representation_module': 'corrflow',
        'representation_checkpoint': '/checkpoint/cinjon/spaceofmotion/supercons/corrflow.kineticsmodel.pth',
        'checkpoint_path': checkpoint_path
    }
    for num_rumn in range(2):
        for do_feat_conversion in [False, True]:
            for tem_step_gamma in [0.75, 0.5]:
                for tem_step_size in [10, 8]:
                    for num_gpus in [8]:
                        for lr in [1e-4, 3e-4]:
                            counter += 1                        
                            _job = {k: v for k, v in job.items()}
                            _job['num_gpus'] = num_gpus
                            _job['tem_training_lr'] = lr
                            _job['tem_step_gamma'] = tem_step_gamma
                            _job['tem_step_size'] = tem_step_size
                            _job['name'] = '%s-%05d' % (_job['name'], counter)
                            _job['num_cpus'] = num_gpus * 10
                            _job['gb'] = 64 * num_gpus
                            _job['tem_batch_size'] = 4 if do_feat_conversion else 1
                            _job['do_feat_conversion'] = do_feat_conversion
                            _job['time'] = 16 if do_feat_conversion else 24
                            
                            if find_counter == counter:
                                return _job
                            
                            # if not find_counter:
                            #     func(_job, counter, email, code_directory)


    # print("Counter: ", counter)  # 368
    # These are doing the full gymnastics dataset, but with augmentation.
    # We skip the non- augment ones that are on do_feat_conversion because those were done before.
    # The others were also done above but did not go for enough epochs because they take a while.
    job = {
        'name': '2019.09.17',
        'video_anno': os.path.join(anno_directory, 'anno_fps12.on.sep052019.json'),
        'video_info': os.path.join(anno_directory, 'video_info.sep052019.fps12.csv'),
        'do_representation': True,
        'module': 'TEM',
        'mode': 'train',
        'tem_compute_loss_interval': 10,
        'num_videoframes': 100,
        'skip_videoframes': 2,
        'tem_batch_size': 4,
        'tem_step_size': 10,
        'tem_step_gamma': 0.5,
        'tem_epoch': 30,
        'tem_train_subset': 'train',
        'do_representation': True,
        'representation_module': 'corrflow',
        'representation_checkpoint': '/checkpoint/cinjon/spaceofmotion/supercons/corrflow.kineticsmodel.pth',
        'checkpoint_path': checkpoint_path
    }
    for do_augment in [True, False]:
        for do_feat_conversion in [False, True]:
            for tem_step_gamma in [0.75, 0.5]:
                for tem_step_size in [10, 8]:
                    for num_gpus in [8]:
                        for lr in [1e-4, 3e-4]:
                            if not do_augment and do_feat_conversion:
                                continue
                            counter += 1
                            
                            _job = {k: v for k, v in job.items()}
                            _job['num_gpus'] = num_gpus
                            _job['tem_training_lr'] = lr
                            _job['tem_step_gamma'] = tem_step_gamma
                            _job['tem_step_size'] = tem_step_size
                            _job['name'] = '%s-%05d' % (_job['name'], counter)
                            _job['num_cpus'] = num_gpus * 10
                            _job['gb'] = 64 * num_gpus
                            _job['tem_batch_size'] = 4 if do_feat_conversion else 1
                            _job['do_feat_conversion'] = do_feat_conversion
                            _job['do_augment'] = do_augment
                            _job['time'] = 16 if do_feat_conversion else 24
                            
                            if find_counter == counter:
                                return _job
                            
                            # if not find_counter:
                            #     func(_job, counter, email, code_directory)


    print("Counter: ", counter)  # 424
    # These are doing the thumos images with augmentation and without, with do_feat_conversion and w/o.
    # These are important because we beleive this is working properly now based on reproduction.
    job = {
        'name': '2019.09.29',
        'video_info': '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/thumos14_annotations',
        'dataset': 'thumosimages',
        'module': 'TEM',
        'mode': 'train',
        'tem_compute_loss_interval': 1,
        'tem_epoch': 30,
        'do_representation': True,
        'num_videoframes': 100,
        'skip_videoframes': 5,
        'representation_module': 'corrflow',
        'representation_checkpoint': '/checkpoint/cinjon/spaceofmotion/supercons/corrflow.kineticsmodel.pth',
        'checkpoint_path': checkpoint_path,
        'tem_nonlinear_factor': 0.1
    }
    for do_augment in [True, False]:
        for do_feat_conversion in [True, False]:
            for tem_milestones in ['5,15', '5,20', '10,25']:
                for tem_step_gamma in [0.1, 0.5]:
                    for tem_l2_loss in [0.01, 0.005, 0.001]:
                        for lr in [1e-4, 1e-3]:
                            counter += 1
                            _job = {k: v for k, v in job.items()}
                            
                            batch_size = 4 if do_feat_conversion else 1
                            num_gpus = min(int(16 / batch_size), 8)
                            _job['tem_batch_size'] = batch_size
                            _job['num_gpus'] = num_gpus
                            _job['tem_training_lr'] = lr
                            _job['tem_lr_milestones'] = tem_milestones
                            _job['name'] = '%s-%05d' % (_job['name'], counter)
                            _job['num_cpus'] = num_gpus * 10
                            _job['gb'] = 64 * num_gpus
                            
                            _job['do_feat_conversion'] = do_feat_conversion
                            _job['do_augment'] = do_augment
                            _job['tem_step_gamma'] = tem_step_gamma
                            _job['tem_l2_loss'] = tem_l2_loss
                            _job['time'] = 16 if do_feat_conversion else 24
                        
                            if find_counter == counter:
                                return _job
                        
                            if not find_counter:
                                func(_job, counter, email, code_directory)
                            
                        
if __name__ == '__main__':
    run()
