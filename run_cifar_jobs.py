import os
import sys


def do_jobarray(counter, job, time, find_counter, do_job=False):
    # code_directory = '/home/resnick/Code/BSN-boundary-sensitive-network.pytorch'
    code_directory = '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch'    
    num_gpus = 1
    num_cpus = num_gpus * 10
    # gb = num_gpus * 16
    gb = num_gpus * 64    
    directory = '/checkpoint/cinjon/spaceofmotion/bsn'
    slurm_logs = os.path.join(directory, 'slurm_logs')
    slurm_scripts = os.path.join(directory, 'slurm_scripts')
    
    comet_dir = os.path.join(directory, 'comet', 'cifar-%d' % job['num_classes'], job['model'].lower())
    if not os.path.exists(comet_dir):
        os.makedirs(comet_dir)        
    job['local_comet_dir'] = comet_dir
    job['time'] = time
    
    job['num_workers'] = min(int(2.5 * num_gpus), num_cpus - num_gpus)
    
    jobarray = []

    model = job['model']
    if model == 'resnet':
        batch_sizes = [256]
    else:
        batch_sizes = [64]
    lrs = [0.3, 0.1, 0.03]
    min_lrs = [0.0003, 0.00003]
    weight_decays = [0.0, 0.0005, 0.00005]
    lr_intervals = [20, 50]
    epochs = 500
    
    for bs in batch_sizes:
        for lr in lrs:
            for min_lr in min_lrs:
                for wd in weight_decays:
                    for lr_int in lr_intervals:
                        for not_pretrain in [True, False]:
                            counter += 1
                            _job = {k: v for k, v in job.items()}
                            _job['counter'] = counter
                            _job['batch_size'] = bs
                            _job['lr'] = lr
                            _job['lr_interval'] = lr_int
                            _job['weight_decay'] = wd
                            _job['min_lr'] = min_lr
                            _job['not_pretrain'] = not_pretrain

                            _job['name'] = '%s.%s-%05d' % (_job['name'], model, counter)
                            if find_counter == counter:
                                return counter, _job
                            jobarray.append(counter)
                                
    if not find_counter and do_job:
        jobname = 'cifar.%s.%dhr.cnt%d' % (model, time, counter)
        jobcommand = "python cifar.py --mode array"
        print("Size: ", len(jobarray), jobcommand, " /.../ ", jobname)

        slurmfile = os.path.join(slurm_scripts, jobname + '.slurm')
        hours = int(time)
        minutes = int((time - hours) * 60)
        with open(slurmfile, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --job-name=%s\n" % jobname)
            f.write("#SBATCH --array=%s\n" % ','.join([str(c) for c in jobarray]))
            f.write("#SBATCH --mail-type=END,FAIL\n")
            f.write("#SBATCH --mail-user=cinjon@nyu.edu\n")
            f.write("#SBATCH --cpus-per-task=%d\n" % num_cpus)
            f.write("#SBATCH --time=%d:%d:00\n" % (hours, minutes))
            f.write("#SBATCH --gres=ntasks-per-node=1\n")
            f.write("#SBATCH --gres=gpu:%d\n" % num_gpus)
            f.write("#SBATCH --mem=%dG\n" % gb)
            f.write("#SBATCH --nodes=%d\n" % 1)
            f.write("#SBATCH --output=%s\n" % os.path.join(
                slurm_logs, jobname + ".%A.%a.out"))
            f.write("#SBATCH --error=%s\n" % os.path.join(
                slurm_logs, jobname + ".%A.%a.err"))

            f.write("module purge" + "\n")
            f.write("module load cuda/10.0\n")            
            f.write("source activate onoff\n")
            f.write("SRCDIR=%s\n" % code_directory)
            f.write("cd ${SRCDIR}\n")
            f.write(jobcommand + "\n")

        s = "sbatch %s" % os.path.join(slurm_scripts, jobname + ".slurm")
        os.system(s)
    return counter, None
    

def do(find_counter=None, do_job=False):
    counter = 0
    
    job = {
        'name': '2019.11.11', 'num_classes': 10
    }
    for model in ['ccc', 'resnet', 'amdim', 'corrflow']:
        _job = job.copy()
        _job['model'] = model
        counter, _job = do_jobarray(counter, _job, time=5, find_counter=find_counter, do_job=False)
        if find_counter and find_counter == counter:
            return counter, _job

        
    job = {
        'name': '2019.11.11.c100', 'num_classes': 100
    }
    for model in ['ccc', 'resnet', 'amdim', 'corrflow']:
        _job = job.copy()
        _job['model'] = model
        counter, _job = do_jobarray(counter, _job, time=3, find_counter=find_counter, do_job=False)
        if find_counter and find_counter == counter:
            return counter, _job


    job = {
        'name': '2019.11.12.c100', 'num_classes': 100
    }
    for model in ['ccc', 'resnet', 'amdim', 'corrflow']:
        _job = job.copy()
        _job['model'] = model
        counter, _job = do_jobarray(counter, _job, time=10, find_counter=find_counter, do_job=False)
        if find_counter and find_counter == counter:
            return counter, _job


    job = {
        'name': '2019.11.12.c100', 'num_classes': 100
    }
    for model in ['ccc', 'resnet', 'corrflow']:
        _job = job.copy()
        _job['model'] = model
        counter, _job = do_jobarray(counter, _job, time=10, find_counter=find_counter, do_job=do_job)
        if find_counter and find_counter == counter:
            return counter, _job
        

if __name__ == '__main__':
    do(do_job=True)
