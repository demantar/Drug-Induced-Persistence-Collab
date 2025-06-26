# utilities to make stochastic simulations given a description of a measurement type
import model_utils as utils
import numpy as np
import random
from random import random as rand
import math

# get a simulation for a measurement type meas_type with a variable parameter
# set par when initially we have z0 cells of type 0 and z1 cells of type 1
# rel_meas_error is the standard deviation of the multiplicative measurement error
# (the measurement is multiplied by exp(x) where x is a sample from a normal distribution
#  centered at zero with std deviation rel_meas_error)
def simulate(par, meas_type, z0, z1, rel_meas_error=0.0):
    data = []
    z0_hist = []
    z1_hist = []
    for schedule in meas_type.doses:
        s_data, (s_z0_hist, s_z1_hist) = simulate_one_schedule(par, meas_type.change_times, 
                                                     meas_type.meas_times, schedule, 
                                                     z0, z1, rel_meas_error)
        data.append(s_data)
        z0_hist.append(s_z0_hist)
        z1_hist.append(s_z1_hist)
    return utils.Measurement(meas_type, np.array(data)), (np.array(z0_hist), np.array(z1_hist))
    

# helper function that simulates using a given dosage schedule
def simulate_one_schedule(par, change_times, meas_times, doses, z0, z1, rel_meas_error=0.0):
    curr_par = utils.get_fund_param_set(par, 0)
    t = 0
    meas_time_ind = 0
    change_time_ind = 0
    res = []
    z0_hist = []
    z1_hist = []
    while meas_time_ind < len(meas_times) and 0 < z0+z1 and z0+z1 < 1000000:
        if t >= meas_times[meas_time_ind]:
            res.append((z0 + z1) * 
                       math.exp(random.gauss(mu=0, 
                                             sigma=rel_meas_error)))
            z0_hist.append(z0)
            z1_hist.append(z1)
            meas_time_ind += 1

        # calculate the rate of events of each type of cells
        event_rate_0 = z0 * (curr_par.b0 + curr_par.d0 + curr_par.mu)
        event_rate_1 = z1 * (curr_par.b1 + curr_par.d1 + curr_par.nu) 
        event_rate_tot = event_rate_0 + event_rate_1
        # calculate time until next event
        dt = - 1/event_rate_tot * math.log(rand())

        # if dosage is changed before next event, change dosage and abort event
        if (change_time_ind < len(change_times) 
                and t + dt >= change_times[change_time_ind]):
            t = change_times[change_time_ind]
            curr_par = utils.get_fund_param_set(par, doses[change_time_ind])
            change_time_ind += 1
            continue
        t += dt

        # go through cases of different types of events
        u = rand()
        if u < event_rate_0 / event_rate_tot: # events for type 0 cell
            v = rand()
            if v < z0 * curr_par.b0 / event_rate_0: 
                z0 += 1 # split
            elif v - z0 * curr_par.b0 / event_rate_0 < z0 * curr_par.d0 / event_rate_0:
                z0 -= 1 # die
            else:
                z0 -= 1 # change type
                z1 += 1
        else: #events for type 1 cell
            v = rand()
            if v < z1 * curr_par.b1 / event_rate_1: 
                z1 += 1 # split
            elif v - z1 * curr_par.b1 / event_rate_1 < z1 * curr_par.d1 / event_rate_1:
                z1 -= 1 # die
            else:
                z1 -= 1 # change type
                z0 += 1

    while meas_time_ind < len(meas_times):
        res.append(-1) # represents no measurement
        z0_hist.append(-1)
        z1_hist.append(-1)
        meas_time_ind += 1
    return res, (z0_hist, z1_hist)




