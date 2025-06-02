# short script to plot a single experment from a pkl file
import fitting_experiments
import model_utils as utils
import pickle
import sys

if len(sys.argv) > 1:
    file = sys.argv[1]
else:
    file = 'data_7.pkl'
if len(sys.argv) > 2:
    run = int(sys.argv[2])
else:
    run = 0

with open(file, 'rb') as f:
    fits, sims, used_params = pickle.load(f)
    fitting_experiments.plot_params_fit_log_growth_pulsed(sims[run], fits[run], used_params)
    fitting_experiments.plot_params_fit_f0_pulsed(sims[run], fits[run], used_params)
    fitting_experiments.plot_params_fit_log_growth_pulsed_diff(sims[run], fits[run], used_params)
