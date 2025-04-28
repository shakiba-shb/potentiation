import argparse
import os
import itertools as it
import time

# NSGA2 best is 6654 and worst is 32400
# lexicase best is 9540 and worst is 24481
# asecual run:
# NSGA2 best is 15860 and worst is 29756
# lexicase best is 26311 and worst is 32400

parser = argparse.ArgumentParser(description="Submit jobs for replays")
parser.add_argument('-snapshotFile', action='store', dest='snapshotFiles', default='alg-NSGA2_S-100_N-100_K-8_emprand-13_seed-15860_snapshot.npz', type=str, help='File containing snapshots.')
parser.add_argument('-replayGen', action='store', dest='replayGens',type=str, default ='50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000', help='The snapshot to replay')
parser.add_argument('-n_gen', action='store', dest='n_gens', type=str, default = '1000', help='Number of generations')
parser.add_argument('-replaySeed', action='store', type=str, dest='replaySeeds', 
                    default='9629,330,4615,905,9366,6230,1661,5981,4550,1044,'
                    '6004,4883,8238,9877,3689,1209,4149,4438,2637,2036,'
                    '2952,8266,8507,8762,2995,5397,5785,1636,759,6980', help='Random seed for replays')
parser.add_argument('-rdir', type=str, default = '/mnt/scratch/shahban1/potentiation_round3/', help='Results directory')
parser.add_argument('-n_trials', action='store', dest='N_TRIALS', default=20, type=int, help='Number of trials to run')
parser.add_argument('-n_jobs', action='store', default=1, type=int, help='Number of parallel jobs')
parser.add_argument('-mem', action='store', dest='mem', default=10000, type=int, help='memory request and limit (MB)')
parser.add_argument('-time', action='store', dest='time', default='2:00:00', type=str, help='time in HR:MN:SS')

args = parser.parse_args()
n_trials = len(args.replaySeeds)
snapshotFiles = args.snapshotFiles.split(',')
replayGens = args.replayGens.split(',')
replaySeeds = args.replaySeeds.split(',')
n_gens = args.n_gens.split(',')
rdir = args.rdir

all_commands = []
job_info = []

for file,replayGen,s,n_gen in it.product(snapshotFiles,replayGens,replaySeeds,n_gens):

    seed =  int(file.split("seed-")[1].split("_")[0])
    all_commands.append(
        f'python /mnt/home/shahban1/potentiation/single_replay_experiment.py -snapshotFile {file} -replayGen {int(replayGen)} -replaySeed {int(s)} -rdir {rdir} -n_gen {int(n_gen)}'
    )

    job_info.append({
        'snapshotFile': file,
		'seed': seed,
        'replayGen': replayGen,
        'replaySeed': s,
        'rdir': rdir
    })

print(len(job_info), 'total jobs created')
time.sleep(3)

jobarrayfile = 'jobfiles/joblist.txt'
os.makedirs('jobfiles', exist_ok=True)
for i, run_cmd in enumerate(all_commands):

    job_name = '_'.join([x + '-' + f'{job_info[i][x]}' for x in ['seed', 'replaySeed', 'replayGen']])
    
    job_file = f'jobfiles/{job_name}.sb'
    
    out_file = job_info[i]['rdir'] + 'replays_out/' + job_name + '_%J.out'

    batch_script = (
    f"""#!/usr/bin/bash 
#SBATCH -A ecode
#SBATCH --output={out_file} 
#SBATCH --job-name={job_name} 
#SBATCH --ntasks={1} 
#SBATCH --cpus-per-task={1} 
#SBATCH --time={args.time}
#SBATCH --mem={args.mem}

date
source /mnt/home/shahban1/potentiation/potenv/bin/activate

{run_cmd}

date
"""
        )

    with open(job_file, 'w') as f:
        f.write(batch_script)

    print(run_cmd)
    # sbatch_response = subprocess.check_output(
    # 	[f'sbatch {job_file}'], shell=True).decode()     # submit jobs
    # print(sbatch_response)
