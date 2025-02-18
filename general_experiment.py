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

def find_best(N, K, emp_random):

    nk = eco.NKLandscape(N, K, eco.Random(emp_random))
    best_fitness = np.zeros(N)

    for i in range(N):
        max_fitness = float('-inf')
        for j in range(2**K):
            fitness = nk.GetFitness(i, j)
            max_fitness = max(max_fitness, fitness)
        best_fitness[i] = max_fitness 

    return best_fitness

class nkProblem(ElementwiseProblem):
    def __init__(self, n_var, n_obj, xl, xu, K, emp_random, **kwargs):

        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = xl
        self.xu = xu
        self.K = K
        self.emp_random = emp_random

        super().__init__(n_var=self.n_var,
                        n_obj=self.n_obj,
                        xl=np.array([self.xl] * self.n_var),
                        xu=np.array([self.xu] * self.n_var),
                        **kwargs
                        )
        
    def _evaluate(self, x, out, *args, **kwargs):

        N = self.n_obj
        K = self.K
        emp_random = self.emp_random
        nk = eco.NKLandscape(N, K, eco.Random(emp_random))
        x = x.astype(int)
        f = np.empty((N))

        for i in range(N):
            neighbors = [x[(i + j) % N] for j in range(1, K + 1)]
            number = int("".join(map(str, neighbors)), 2)
            f[i] = nk.GetFitness(i, number)

        out["F"] = -f



N = 5
K = 3
emp_random = 13
n_var = N
n_obj = N

# nk = eco.NKLandscape(N, K, eco.Random(emp_random))
# for i in range(N):
#     for j in range(2**K):
#         print(i, j, nk.GetFitness(i,j))

problem = nkProblem(n_var = N, n_obj = N, xl = 0, xu = 1, K = K, emp_random = emp_random)
algorithm = Lexicase(
    pop_size=10,
    sampling=BinaryRandomSampling(),
    crossover=TwoPointCrossover(),
    mutation=BitflipMutation(),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 100)

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

X = res.X # the final population (genotypes)
F = res.F # the final population (phenotypes)
best_fitness = find_best(N, K, emp_random)

print(F)
print(best_fitness) 

