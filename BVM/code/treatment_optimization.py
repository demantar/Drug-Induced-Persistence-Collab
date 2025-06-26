# Various functions that find the optimal treatment given certain parameter sets
import model_utils as utils
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import ClassVar, Sequence, Tuple
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
    def get_rate(self, strategy_params: Sequence[float], model_params) -> Tuple[float, float]:
        """
        Get the long term growth rate given certain 
        strategy parameters and model parameters.
        Also returns the f0 in which the strategy behaves optimally.
        """
        pass

    def optimize_rate(self, model_params) -> Tuple[float, Sequence[float], float]:
        """
        Find the strategy parameters that minimize 
        get_rate given specific model parameters.
        Returns a tuple with:
            - the long term rate
            - the paraeters that achieve the rate
            - the equilibrium f0 under the strategy
        """
        obj = lambda strategy_params: self.get_rate(strategy_params, model_params)[0]

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

        rate, f0_inf = self.get_rate(res.x, model_params)

        return rate, res.x, f0_inf
        
@dataclass(slots=True)
class FixedDose(LongTermStrategy):
    """
    Strategy that administers a constant dose.
    The bounds on the dose must be provided.
    """
    no_params: int = 1
    param_names: list[str] = field(default_factory=lambda: ['c'])

    def get_rate(self, strategy_params: Sequence[float], model_params) -> Tuple[float, float]:
        c = strategy_params[0]
        params = utils.get_fund_param_set(model_params, c)
        f0_inf, rate = utils.calc_equilib(params)
        return rate, f0_inf[0]
        
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

    def get_rate(self, strategy_params: Sequence[float], model_params) -> Tuple[float, float]:
        c = strategy_params[0]
        phi = strategy_params[1]
        params_on = utils.get_fund_param_set(model_params, c)
        params_off = utils.get_fund_param_set(model_params, 0)
        params_comb = utils.FundamentalParamSet(
            *(phi * p_on + (1 - phi) * p_off 
              for p_on, p_off 
              in zip(params_on, params_off))
        )
        f0_inf, rate = utils.calc_equilib(params_comb)
        return rate, f0_inf[0]


