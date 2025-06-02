import model_utils as utils
import numpy as np
import random
from random import random as rand
import math

# function that takes a parameter regime, a utils.MeasurementType tuple
# and initial population sizes and returns a utils.Measurement object 
def simulate_fixed(par, meas_type, z0, z1, rel_meas_error=0.0):
    return simulate_pulsed(
                par, utils.meas_type_to_meas_type_pulsed(meas_type), 
                z0, z1, rel_meas_error=0.0
            )

def simulate_pulsed(par, meas_type, z0, z1, rel_meas_error=0.0):
    data = []
    for schedule in meas_type.doses:
        data.append(simulate_one_schedule(par, meas_type.change_times, meas_type.meas_times, schedule, 
                                     z0, z1, rel_meas_error))
    return utils.Measurement(meas_type, np.array(data))
    

# helper function that simulates using a given dosage schedule
def simulate_one_schedule(par, change_times, meas_times, doses, z0, z1, rel_meas_error=0.0):
    curr_par = utils.get_fund_param_set_bd(par, 0)
    t = 0
    meas_time_ind = 0
    change_time_ind = 0
    res = []
    while meas_time_ind < len(meas_times) and z0 + z1 > 0: # TODO: what if z0 + z1 == 0
        if t >= meas_times[meas_time_ind]:
            res.append((z0 + z1) * 
                       math.exp(random.gauss(mu=0, 
                                             sigma=rel_meas_error)))
            meas_time_ind += 1

        event_rate_0 = z0 * (curr_par.b0 + curr_par.d0 + curr_par.mu)
        event_rate_1 = z1 * (curr_par.b1 + curr_par.d1 + curr_par.nu) 
        event_rate_tot = event_rate_0 + event_rate_1
        dt = - 1/event_rate_tot * math.log(rand())

        if (change_time_ind < len(change_times) 
                and t + dt >= change_times[change_time_ind]):
            t = change_times[change_time_ind]
            curr_par = utils.get_fund_param_set_bd(par, doses[change_time_ind])
            change_time_ind += 1
            continue
        t += dt

        u = rand()
        if u < event_rate_0 / event_rate_tot:
            v = rand()
            if v < z0 * curr_par.b0 / event_rate_0:
                z0 += 1
            elif v - z0 * curr_par.b0 / event_rate_0 < z0 * curr_par.d0 / event_rate_0:
                z0 -= 1
            else:
                z0 -= 1
                z1 += 1
        else:
            v = rand()
            if v < z1 * curr_par.b1 / event_rate_1:
                z1 += 1
            elif v - z1 * curr_par.b1 / event_rate_1 < z1 * curr_par.d1 / event_rate_1:
                z1 -= 1
            else:
                z1 -= 1
                z0 += 1
    return res




