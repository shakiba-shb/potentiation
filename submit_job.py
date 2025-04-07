import argparse
import os
import itertools as it
import time


parser = argparse.ArgumentParser(description="Submit jobs")
parser.add_argument('-alg_name', action='store', dest='algs', default='NSGA2', type=str, help='Algorithm name')
parser.add_argument('-S', action = 'store', dest='Ss', type=str, default = '100', help='Population size')
parser.add_argument('-N', action = 'store', dest='Ns', type=str, default = '50,100', help='N in NK landscape')
parser.add_argument('-K', action = 'store', dest='Ks', type=str, default = '8,10,12', help='K in NK landscape')
parser.add_argument('-n_gen', action = 'store', dest='n_gens', type=str, default = '1000', help='Number of generations')
parser.add_argument('-seed', action = 'store', dest='seeds', type=str, default = '14724,24284,31658,6933,1318,16695,27690,8233,24481,6832,13352,4866,12669,12092,15860,19863,6654,10197,29756,14289', help='Random seed')
parser.add_argument('-rdir', type=str, default = '/mnt/scratch/shahban1/potentiation_round1/', help='Results directory')
parser.add_argument('-n_trials', action='store', dest='N_TRIALS', default=20, type=int, help='Number of trials to run')
parser.add_argument('-n_jobs', action='store', default=1, type=int, help='Number of parallel jobs')
parser.add_argument('-mem', action='store', dest='mem', default=10000, type=int, help='memory request and limit (MB)')
parser.add_argument('--slurm', action='store_true', default=False, help='Run on an slurm HPC')
parser.add_argument('-time', action='store', dest='time', default='10:00:00', type=str, help='time in HR:MN:SS')
args = parser.parse_args()

n_trials = len(args.seeds)
seeds = args.seeds.split(',')[:n_trials]
algs = args.algs.split(',')
Ss = args.Ss.split(',')
Ns = args.Ns.split(',')
Ks = args.Ks.split(',')
n_gens = args.n_gens.split(',')
args.slurm = True

# write run commands
all_commands = []
job_info = []
rdir = '/'.join([args.rdir])
os.makedirs(rdir, exist_ok=True)

for alg,s,n,k,seed,n_gen in it.product(algs,Ss,Ns,Ks,seeds,n_gens):

	all_commands.append(
		f'python /mnt/home/shahban1/potentiation/single_nk_experiment.py -alg_name {alg} -S {int(s)} -N {int(n)} -K {int(k)} -seed {int(seed)} -rdir {rdir} -n_gen {int(n_gen)}'
	)

	job_info.append({
		'alg':alg,
        'S':s,
        'N':n,
        'K':k,  
		'seed': seed,
		'rdir': rdir
	})
	
print(len(job_info), 'total jobs created')

time.sleep(3)
if args.slurm:
	# write a jobarray file to read commans from
	jobarrayfile = 'jobfiles/joblist.txt'
	os.makedirs('jobfiles', exist_ok=True)
	for i, run_cmd in enumerate(all_commands):

		job_name = '_'.join([x + '-' + f'{job_info[i][x]}' for x in
							['alg','S','K', 'N', 'seed']])
		job_file = f'jobfiles/{job_name}.sb'
		out_file = job_info[i]['rdir'] + '/' + job_name + '_%J.out'

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

