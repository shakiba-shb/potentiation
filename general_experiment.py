from ec_ecology_toolbox import selection_probabilities as eco
import random
from pymoo.algorithms.moo.nsga2 import NSGA2
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo_lexicase import Lexicase
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import Hypervolume
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
from nkLandscapes_pymoo import nkProblem, find_best


N = 10
K = 5
emp_random = 13
n_var = N
n_obj = N

# nk = eco.NKLandscape(N, K, eco.Random(emp_random))
# for i in range(N):
#     for j in range(2**K):
#         print(i, j, nk.GetFitness(i,j))

problem = nkProblem(n_var = N, n_obj = N, xl = 0, xu = 1, K = K, emp_random = emp_random)
algorithm = NSGA2(
    pop_size=100,
    sampling=BinaryRandomSampling(),
    crossover=TwoPointCrossover(),
    mutation=BitflipMutation(),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 10)

res = minimize(problem,
               algorithm,
               termination,
               seed=10,
               save_history=True,
               verbose=True)

X = res.X.astype(int) # the final population (genotypes)
F = -res.F # the final population (phenotypes)
opt_X = res.opt.get("X").astype(int) # the final solutions genotypes (pf)
opt_F = -res.opt.get("F") # the final solutions phenotypes (pf)

best_fitness = find_best(N, K, emp_random)
scores = np.max(F, axis=0)
matches = (scores == best_fitness)  
num_matches = np.sum(matches)
#coverage = np.sum(np.max(F, axis=0) > best_fitness)
coverage = 0

for i in range(N):
    if (np.max(F, axis=0)[i] >= best_fitness[i]):
        coverage += 1
    
# ref_point = np.array([1]*N)
# ind = Hypervolume(pf = opt_F, ref_point=ref_point)
# hv = ind._do(opt_F)

print(X)
print(F)
print(best_fitness) 
print("score: ", num_matches)
print("coverage: ", coverage)   
# print("hv: ", hv)


# # import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
# df = pd.DataFrame(F)

# # Create the boxplot
# plt.figure(figsize=(8, 6))
# sns.boxplot(data=df)

# means = df.mean()
# #plt.scatter(range(len(means)), means, color='black', marker='o', label='Mean')
# plt.scatter(range(len(best_fitness)), best_fitness, color='red', marker='*', s=150, label='Best Fitness')

# # Labels and title
# plt.xlabel("N (objective)")
# plt.ylabel("Fitness")
# plt.title(f"Fitness of each objective in the final population\n N = {N}, K = {K}")
# plt.legend()
# plt.show()