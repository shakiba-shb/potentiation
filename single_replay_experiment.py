import uuid
import numpy as np
import json
import argparse
import os
import re

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

def experiment (snapshotFile = None, replayGen = None, n_gen = None, replaySeed = None, rdir = ""):

    runid = uuid.uuid4()
    pattern = r"alg-(?P<alg>.+)_S-(?P<S>\d+)_N-(?P<N>\d+)_K-(?P<K>\d+)_emprand-(?P<emprand>\d+)_seed-(?P<seed>\d+)_snapshot.npz"
    
    match = re.search(pattern, snapshotFile)
    if match:
        alg_name = match.group("alg")
        S = int(match.group("S"))
        N = int(match.group("N"))
        K = int(match.group("K"))
        emprand = int(match.group("emprand"))
        seed = int(match.group("seed"))

        init_data = np.load(rdir+alg_name+'/snapshots/'+snapshotFile)
        init_pop = init_data[f'init_pop_{replayGen}']


    #Define the problem
    problem = nkProblem(n_var = N, n_obj = N, xl = 0, xu = 1, K = K, emp_random = emprand)

    #Define the algorithms
    if alg_name == "lex_std":
        algorithm = Lexicase(
            pop_size=S,
            sampling=init_pop,
            crossover=SBX(prob=0, prob_var=0),
            mutation=BitflipMutation(),
            eliminate_duplicates=True
        )

    elif alg_name == "NSGA2":
        algorithm = NSGA2(
            pop_size=S,
            sampling=init_pop,
            crossover=SBX(prob=0, prob_var=0),
            mutation=BitflipMutation(),
            eliminate_duplicates=True
        )

    termination = get_termination("n_gen", n_gen)

    res = minimize(problem,
                algorithm,
                termination,
                seed=replaySeed,
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
    folder_name = rdir+f'{alg_name}/replays'
    os.makedirs(folder_name, exist_ok=True)
    np.savez(folder_name+f'/alg-{alg_name}_S-{S}_N-{N}_K-{K}_emprand-{emprand}_seed_{seed}_replaySeed-{replaySeed}_replayGen_{replayGen}.npz', X=X, F=F, 
            opt_X=opt_X, opt_F=opt_F, best_fitness=best_fitness)
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run a single comparison experiment')
    parser.add_argument('-snapshotFile', type=str, default = 'alg-NSGA2_S-100_N-100_K-8_emprand-13_seed-32400_snapshot.npz', help='File containing snapshots.')
    parser.add_argument('-replayGen', type=int, default = 100, help='The snapshot to replay')
    parser.add_argument('-n_gen', type=int, default = 100, help='Number of generations')
    parser.add_argument('-replaySeed', type=int, default = 0, help='Random seed for replays')
    parser.add_argument('-rdir', type=str, default = '/home/shakiba/potentiation/results/', help='Results directory')
    args = parser.parse_args()

    experiment(snapshotFile = args.snapshotFile, replayGen = args.replayGen, n_gen = args.n_gen, replaySeed = args.replaySeed, rdir = args.rdir)


