# Various functions that find the optimal treatment given certain parameter sets
import model_utils as utils
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import ClassVar, Sequence
import random
import scipy.optimize
import numpy as np
import plotly.express as px

@dataclass(slots=True)
class LongTermStrategy(ABC):
    """Abstract class for dosing strategies"""
    param_bounds: list[tuple[float, float]]
    param_names: list[str]
    no_params: int

    @abstractmethod
    def get_rate(self, strategy_params: Sequence[float], model_params) -> float:
        """
        Get the long term growth rate given certain 
        strategy parameters and model parameters
        """
        pass

    def optimize_rate(self, model_params) -> tuple[float, Sequence[float]]:
        """
        Find the strategy parameters that minimize 
        get_rate given specific model parameters
        """
        obj = lambda strategy_params: self.get_rate(strategy_params, model_params)
        x0 = [random.uniform(lo, hi) for lo, hi in self.param_bounds]
        res = scipy.optimize.minimize(
            obj, 
            x0, 
            bounds=self.param_bounds, 
            method='L-BFGS-B',
            options={
                'ftol': 1e-12,     
                'gtol': 1e-10,     
                'maxiter': 1000,  
            }
        )

        if not res.success:
            print(f"Strategy optimization: {res.message}")
        return res.fun, res.x
        
@dataclass(slots=True)
class FixedDose(LongTermStrategy):
    """
    Strategy that administers a constant dose.
    The bounds on the dose must be provided.
    """
    no_params: int = 1
    param_names: list[str] = field(default_factory=lambda: ['c'])

    def get_rate(self, strategy_params: Sequence[float], model_params) -> float:
        c = strategy_params[0]
        params = utils.get_fund_param_set(model_params, c)
        _, rate = utils.calc_equilib(params)
        return rate
        
@dataclass(slots=True)
class RapidOnOffPulse(LongTermStrategy):
    """
    Strategy that administers a constant dose and no dose
    in rapid succession. The dose administered is c and 
    the proportion of the total time with the dose on is phi.
    The bounds on c and phi must be provided.
    """
    no_params: int = 2
    param_names: list[str] = field(default_factory=lambda: ['c', 'phi'])

    def get_rate(self, strategy_params: Sequence[float], model_params) -> float:
        c = strategy_params[0]
        phi = strategy_params[1]
        params_on = utils.get_fund_param_set(model_params, c)
        params_off = utils.get_fund_param_set(model_params, 0)
        params_comb = utils.FundamentalParamSet(
            *(phi * p_on + (1 - phi) * p_off 
              for p_on, p_off 
              in zip(params_on, params_off))
        )
        _, rate = utils.calc_equilib(params_comb)
        return rate

if __name__ == '__main__':
    print("test output for treatment_optimization")
    strat_fixed = FixedDose(param_bounds=[(0, 100)])
    strat_pulsed = RapidOnOffPulse(param_bounds=[(0, 100), (0, 1)])
    
    new_param_test = utils.ParamSet_MMmu(
        mu = 0.0004,
        d_mu = 0.004,
        e_mu = 10.0,
        nu = 0.004,
        b0 = 0.04,
        d0 = 0,
        d_d0 = 0.08,
        b1 = 0.001,
        d1 = 0
    )

    print("testing fixed dose strategies")
    strat = strat_fixed
    opt_f, opt_x = strat.optimize_rate(new_param_test)
    print(f'opt_f: {opt_f}, opt_x: {opt_x}')
    doses = np.linspace(0, 100, 300)
    rate_arr = [strat.get_rate([c], new_param_test) for c in doses]
    fig = px.line(x = doses, y = rate_arr)
    fig.show()

    print("testing pulsed strategies")
    strat = strat_pulsed
    opt_f, opt_x = strat.optimize_rate(new_param_test)
    print(f'opt_f: {opt_f}, opt_x: {opt_x}')
    doses = np.linspace(0, 100, 300)
    rate_arr_c = [strat.get_rate([c, opt_x[1]], new_param_test) for c in doses]
    fig = px.line(x = doses, y = rate_arr_c)
    fig.show()
    ratios = np.linspace(0, 1, 100)
    rate_arr_phi = [strat.get_rate([opt_x[0], phi], new_param_test) for phi in ratios]
    fig = px.line(x = ratios, y = rate_arr_phi)
    fig.show()


