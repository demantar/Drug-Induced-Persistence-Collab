import fitting_experiments
import model_utils as utils
import pickle

file = 'data_1.pkl'
run = 0

with open(file, 'rb') as f:
    fits, sims = pickle.load(f)
    fitting_experiments.plot_params_fit_log_growth_pulsed(sims[0], fits[0])
