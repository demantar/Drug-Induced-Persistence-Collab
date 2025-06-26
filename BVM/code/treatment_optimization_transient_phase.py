from typing import NamedTuple
import model_utils as utils
import scipy.optimize
import random
import numpy as np

class TransientPhaseOptimizer:
    """
    A class that stores information on a specific instance of the model
    and allows calculations of various information on the transient phase.
    The model assumes that once f0 reaches f0_final at time t_final a long 
    term strategy such that the total growth during that strategy can be
    accurately approximated via it's average growth rate rho_final.

    Assumptions: (IMPORTANT!)
    * f0_init > f0_final
    * f0' is decreasing in c, 
          (this holds for example if lambda0 is decreasingin c, 
           lambda1 is constant in c, mu is increasing in c and 
           nu is decreasing in c)
    """
    f0_init:               float 
    f0_final:              float
    rho_final:             float
    model_params:          NamedTuple
    c_max:                 float
    f0_resolution:         int
    f0_points:             list[float] 
    dose_f0:               list[float] 
    rate_f0:               list[float]
    time_points:           list[float]
    transient_growth:      float
    trajectory_calculated: bool

    def __init__(self, model_params:  NamedTuple, 
                       f0_init:       float, 
                       f0_final:      float, 
                       rho_final:     float, 
                       max_dose:      float, 
                       f0_resolution: int):
        assert(f0_init > f0_final)
        self.f0_init = f0_init
        self.f0_final = f0_final
        self.rho_final = rho_final
        self.model_params = model_params
        self.c_max = max_dose
        self.f0_resolution = f0_resolution
        self.f0_points = list(np.linspace(f0_final, f0_init, f0_resolution + 2))[1:-1] 
        self.trajectory_calculated = False

    def transient_objective(self, f0: float, c: float) -> float:
        """
        Finds the transient objective (rho(c, f0) - rho_final)/|f0'(f0, c)|
        IMPORTANT: Assumes f0' is negative
        """
        curr_rho = utils.rho_c(f0, self.model_params, c) 
        rho_diff = curr_rho - self.rho_final
        f0_prime = utils.f0_prime_c(f0, self.model_params, c)
        objective = rho_diff / abs(f0_prime)
        return objective

    # TOOD: add option for basin hop / multiple starts
    def optimal_dose(self, f0) -> tuple[float, float]: 
        """
        Finds the best dose to administer when the sensitive cell proportion
        is the given f0. Returns both the optimal dose and the 
        "transient-objective" of that cost
        """

        # find bounds
        assert(f0 >= self.f0_final)
        assert(f0 <= self.f0_init)

        # find lower and upper bound of doses that can steer f0 to f0_final
        f0_prime_c = lambda c: utils.f0_prime_c(f0, self.model_params, c)
        if f0_prime_c(0) < 0:
            c_lb, c_ub = 0, self.c_max
        elif f0_prime_c(self.c_max) > 0:
            raise Exception("not possible to steer into final state")
        else:
            c_lb = scipy.optimize.root_scalar(f0_prime_c, 
                                              bracket=[0, self.c_max]).root
            c_ub = self.c_max

        # find minimum 
        obj = lambda c: self.transient_objective(f0, c)
        x0 = [random.uniform(0, self.c_max)]
        res = scipy.optimize.minimize(
            obj, 
            x0, 
            bounds=[(c_lb + 1e-9, c_ub - 1e-9)], 
            method='L-BFGS-B',
            options={
                'ftol': 1e-12,     
                'gtol': 1e-10,     
                'maxiter': 1000,  
            }
        )

        print(res.x[0], c_lb, c_ub)

        return res.x[0], res.fun

    def calculate_trajectory(self) -> None:
        """
        Fills the variables for the trajectory. That is dose_f0, rate_f0
        """
        self.dose_f0 = []
        self.rate_f0 = []
        self.transient_growth = 0
        for f0 in self.f0_points:
            dose, rate = self.optimal_dose(f0)
            print(dose)
            self.rate_f0.append(rate)
            self.dose_f0.append(dose)
            self.transient_growth += rate

        t = 0
        self.time_points = []
        prev_c = self.dose_f0[0]
        prev_f0 = self.f0_init
        prev_f0_prime = abs(utils.f0_prime_c(prev_f0, self.model_params, prev_c))
        for curr_f0, curr_c in zip(reversed(self.f0_points), reversed(self.dose_f0)):
            print(curr_c)
            curr_f0_prime = abs(utils.f0_prime_c(curr_f0, self.model_params, curr_c))
            delta_f0 = abs(curr_f0 - prev_f0)
            print(prev_f0_prime, curr_f0_prime)
            t += (delta_f0/prev_f0_prime + delta_f0/curr_f0_prime) / 2
            self.time_points.append(t)
            prev_c, prev_f0, prev_f0_prime = curr_c, curr_f0, curr_f0_prime

        self.time_points.reverse()

        self.trajectory_calculated = True




        
        
