import uuid
import numpy as np
import json
import argparse
import os

from ec_ecology_toolbox import selection_probabilities as eco
from pymoo.algorithms.moo.nsga2 import NSGA2
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo_lexicase import Lexicase
from nkLandscapes_pymoo import nkProblem
from pymoo.indicators.hv import Hypervolume

from nkLandscapes_pymoo import nkProblem, find_best

def experiment (alg_name = None, S = None, N = None, K = None, emprand = None, n_gen = None, snapshot = None, seed = None, rdir = ""):

    runid = uuid.uuid4()
    print("alg_name = ", alg_name, "S = ", S, "n_var = ", N, "n_obj = ", N, "K", K, "emprand", emprand, "n_gen = ", n_gen,
         "seed = ", seed, "snapshot", snapshot, "rdir = ", rdir, "runid = ", runid)
    

    #Define the problem
    problem = nkProblem(n_var = N, n_obj = N, xl = 0, xu = 1, K = K, emp_random = emprand)

    #Define the algorithms
    if alg_name == "lex_std":
        algorithm = Lexicase(
            pop_size=S,
            sampling=BinaryRandomSampling(),
            crossover=SBX(prob=0, prob_var=0),
            mutation=BitflipMutation(),
            eliminate_duplicates=True
        )

    elif alg_name == "NSGA2":
        algorithm = NSGA2(
            pop_size=S,
            sampling=BinaryRandomSampling(),
            crossover=SBX(prob=0, prob_var=0),
            mutation=BitflipMutation(),
            eliminate_duplicates=True
        )

    termination = get_termination("n_gen", n_gen)

    res = minimize(problem,
                algorithm,
                termination,
                seed=seed,
                save_history=True,
                verbose=True)
    
    np.set_printoptions(precision=2, suppress=True)
    X = res.pop.get('X').astype(int) #final genotypes
    F = -res.pop.get('F') #final phenotypes
    opt_X = res.opt.get("X").astype(int) #final solutions genotypes (pf)
    opt_F = -res.opt.get("F") #final solutions phenotypes (pf)

    assert len(X) == len(F), "X and F should have the same length"
    assert len(opt_X) == len(opt_F), "opt_X and opt_F should have the same length"

    ###### Hypervolume ######
    # ref_point = np.array([0]*N)
    # ind = Hypervolume(pf = opt_F, ref_point=ref_point)
    # hv = ind._do(opt_F)

    best_fitness = find_best(N, K, emprand)
    #final_population_data = {'X': X.tolist(), 'F': F.tolist(), 'opt_X': opt_X.tolist(), 'opt_F': opt_F.tolist()}
    folder_name = rdir+f'{alg_name}'
    # If snapshot is one, save the population. 
    if snapshot == True:
        snapshots = {}
        for i in range(1, int(int(n_gen)/50) + 1):
            gen = 50 * i
            snapshots[f'init_pop_{gen}'] = res.history[gen-1].pop.get('X')

        new_folder = folder_name + '/snapshots'
        os.makedirs(new_folder, exist_ok=True)
        np.savez(new_folder+f'/alg-{alg_name}_S-{S}_N-{N}_K-{K}_emprand-{emprand}_seed-{seed}_snapshot.npz',
                 **snapshots)
    elif snapshot == False:
        os.makedirs(folder_name, exist_ok=True)
        np.savez(folder_name+f'/alg-{alg_name}_S-{S}_N-{N}_K-{K}_emprand-{emprand}_seed-{seed}.npz', X=X, F=F, 
             opt_X=opt_X, opt_F=opt_F, best_fitness=best_fitness)
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run a single comparison experiment')
    parser.add_argument('-alg_name', type=str, default = 'NSGA2', help='Algorithm to use')
    parser.add_argument('-S', type=int, default = 100, help='Population size')
    parser.add_argument('-N', type=int, default = 50, help='Number of objectives/variables')
    parser.add_argument('-K', type=int, default = 8, help='Number of adjacent bits to look at')
    parser.add_argument('-emprand', type=int, default = 13, help='Random seed for the landscape')
    parser.add_argument('-n_gen', type=int, default = 100, help='Number of generations')
    parser.add_argument('-seed', type=int, default = 0, help='Random seed')
    parser.add_argument('--snapshot', type=bool, default=False, help='Take snapshopts of population')
    parser.add_argument('-rdir', type=str, default = '/mnt/scratch/shahban1/potentiation_round3/', help='Results directory')
    args = parser.parse_args()

    experiment(alg_name = args.alg_name, S = args.S, N = args.N, K = args.K, emprand= args.emprand, n_gen = args.n_gen,
                seed = args.seed, snapshot = args.snapshot, rdir = args.rdir)


