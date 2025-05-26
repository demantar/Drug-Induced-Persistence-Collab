import fitting_experiments
import model_utils as utils
import pickle

file = 'data_1.pkl'
run = 9

with open(file, 'rb') as f:
    fits, sims, used_params = pickle.load(f)
    fitting_experiments.plot_params_fit_log_growth_pulsed(sims[0], fits[0], used_params)
    fitting_experiments.plot_params_fit_f0_pulsed(sims[0], fits[0], used_params)
