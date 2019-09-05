"""Run train on the cluster."""
import os

# NOTE:
# email should be your email. name should be identifying, like `cinjon`.
# code_directory is the directory containing train_video_cycle_simple.py, relative to ${HOME}.
# Examples:
# email = 'cinjon@nyu.edu'
# name = 'cinjon'
# code_directory = 'Code/cycle-consistent-supervision'
# from local_config import email, name, code_directory


def _ensure_dirs(slurm_logs, slurm_scripts):
    for d in [slurm_logs, slurm_scripts]:
        if not os.path.exists(d):
            os.makedirs(d)


def _is_float(v):
    try:
        v = float(v)
        return True
    except:
        return False


def _is_int(v):
    try:
        return int(v) == float(v)
    except:
        return False


def _run_batch(job,
               counter,
               slurm_logs,
               slurm_scripts,
               module_load,
               directory,
               email,
               code_directory,
               local_comet_dir=None):
    _ensure_dirs(slurm_logs, slurm_scripts)

    time = job.get('time', 16)
    hours = int(time)
    minutes = int((time - hours) * 60)
    
    if local_comet_dir:
        job['local_comet_dir'] = local_comet_dir

    num_gpus = job['num_gpus']
    num_cpus = job.pop('num_cpus')
    job['data_workers'] = min(int(2.5 * num_gpus), num_cpus - num_gpus)
    job['data_workers'] = max(job['data_workers'], 12)

    gb = job.pop('gb')
    memory_per_node = min(gb, 500)

    flagstring = " --counter %d" % counter
    for key, value in sorted(job.items()):
        if type(value) == bool:
            if value == True:
                flagstring += " --%s" % key
        elif _is_int(value):
            flagstring += ' --%s %d' % (key, value)
        elif _is_float(value):
            flagstring += ' --%s %.6f' % (key, value)
        else:
            flagstring += ' --%s %s' % (key, value)

    if job['mode'] == 'train':
        jobname = "temtr.%s" % job['name']
    elif job['mode'] == 'inference':
        jobname = "teminf.%s" % job['name']
        
    jobcommand = "python main.py %s" % flagstring
    print(jobcommand)

    slurmfile = os.path.join(slurm_scripts, jobname + '.slurm')

    with open(slurmfile, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=%s\n" % jobname)
        f.write("#SBATCH --mail-type=END,FAIL\n")
        f.write("#SBATCH --mail-user=%s\n" % email)
        f.write("#SBATCH --gres=ntasks-per-node=1\n")
        f.write("#SBATCH --cpus-per-task=%d\n" % num_cpus)
        f.write("#SBATCH --time=%d:%d:00\n" % (hours, minutes))
        f.write("#SBATCH --gres=gpu:%d\n" % num_gpus)
        f.write("#SBATCH --mem=%dG\n" % memory_per_node)
        f.write("#SBATCH --nodes=1\n")

        f.write("#SBATCH --output=%s\n" %
                os.path.join(slurm_logs, jobname + ".out"))
        f.write("#SBATCH --error=%s\n" %
                os.path.join(slurm_logs, jobname + ".err"))

        f.write("module purge" + "\n")
        module_load(f)
        f.write("source activate onoff\n")
        f.write("SRCDIR=%s\n" % code_directory)
        f.write("cd ${SRCDIR}\n")
        f.write(jobcommand + "\n")

    s = "sbatch %s" % os.path.join(slurm_scripts, jobname + ".slurm")
    os.system(s)


def fb_run_batch(job, counter, email, code_directory):

    def module_load(f):
        f.write("module load cuda/10.0\n")

    directory = '/checkpoint/cinjon/spaceofmotion'
    slurm_logs = os.path.join(directory, 'bsn', 'slurm_logs')
    slurm_scripts = os.path.join(directory, 'bsn', 'slurm_scripts')
    comet_dir = os.path.join(directory, 'bsn', 'comet')
    if 'checkpoint_path' not in job:
        checkpoint_path = os.path.join(directory, 'bsn', 'checkpoint')
        job['checkpoint_path'] = checkpoint_path
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        
    _run_batch(job,
               counter,
               slurm_logs,
               slurm_scripts,
               module_load,
               directory,
               email,
               code_directory,
               local_comet_dir=comet_dir)
