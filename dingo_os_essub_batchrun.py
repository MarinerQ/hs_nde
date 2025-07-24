

import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import pickle
import time
import sys
import yaml
import pycbc.psd
import bilby
#import torch 
import json
from copy import deepcopy
import pandas as pd
from pesummary.utils.utils import jensen_shannon_divergence
import logging
#from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

from dingo.core.models.posterior_model import PosteriorModel
from dingo.gw.inference.gw_samplers import GWSampler, GWSamplerGNPE
from dingo.gw.injection import Injection
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.core.result import plot_corner_multi
import corner

PARAMETER_NAMES_14D = ['chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'theta_jn', 'luminosity_distance', 'ra', 'dec', 'psi', 'geocent_time' ]


LATEX_LABEL_DICT = {
	'chirp_mass': r'$\mathcal{M}$',
	'mass_ratio': r'$q$',
	'a_1': r'$a_1$',
	'a_2': r'$a_2$',
	'tilt_1': r'$\theta_1$',
	'tilt_2': r'$\theta_2$',
	'phi_12': r'$\phi_{12}$',
	'phi_jl': r'$\phi_{\mathrm{JL}}$',
	'theta_jn': r'$\theta_{\mathrm{JN}}$',
	'luminosity_distance': r'$d_L$',
	'ra': r'$\alpha$',
	'dec': r'$\delta$',
	'geocent_time': r'$t_c$',
	'psi': r'$\psi$',
	'phase': r'$\phi$',
}

def make_corner_plot(samples_list, injection_parameters, labels, colors, filename, figsize=(20,20), latex_label_dict = LATEX_LABEL_DICT):	
	# Set up the figure
	fig = plt.figure(figsize=figsize)
	
	# Get latex labels
	param_labels = [latex_label_dict[param] for param in injection_parameters.keys()]
	parameters = [param for param in injection_parameters.keys()]
	
	# Get samples and quantiles from each result
	samples_list_to_plot = []
	quantiles_list = []
	for samples in samples_list:
		#samples = result.posterior[parameters].values
		#samples_list.append(samples)
		temp_samples = {param: samples[param] for param in injection_parameters.keys() if param in samples}
		samples_list_to_plot.append(pd.DataFrame(temp_samples))
		
		# Calculate quantiles for each parameter
		quantiles = {}
		for param in injection_parameters.keys():
			q = np.percentile(samples[param], [5, 50, 95])
			quantiles[param] = q
		quantiles_list.append(quantiles)
	
	# Get truth values
	truths = [injection_parameters[param] for param in injection_parameters.keys()]
	
	# Make the corner plot
	figure = corner.corner(
		samples_list_to_plot[0],
		labels=param_labels,
		fig=fig,
		color=colors[0],
		truths=truths,
		truth_color='r',
		show_titles=False,
		labelpad=0.2,
		plot_datapoints=False,
		fill_contours=False,
		grid=False,
		label_kwargs={'fontsize': 16},  
		hist_kwargs={'density': True, 'linewidth': 2.0}, # Increased line width for 1D histograms
		contour_kwargs={'linewidths': 2.0}, # Increased line width for 2D contours
		quantiles=[0.05, 0.95],
		levels=(0.68, 0.95),
		use_math_text=True,
		no_fill_contours=True # Only show contour lines, no filled regions
	)
	
	# Add additional results
	for samples, label, color in zip(samples_list_to_plot[1:], labels[1:], colors[1:]):
		corner.corner(
			samples,
			labels=param_labels,
			fig=fig,
			color=color,
			truths=truths,
			truth_color='r',
			show_titles=False,
			labelpad=0.2,
			plot_datapoints=False,
			fill_contours=False,
			grid=False,
			label_kwargs={'fontsize': 16},  
			hist_kwargs={'density': True, 'linewidth': 2.0}, # Increased line width for 1D histograms
			contour_kwargs={'linewidths': 2.0}, # Increased line width for 2D contours
			quantiles=[0.05, 0.95],
			levels=(0.68, 0.95),
			use_math_text=True,
			no_fill_contours=True # Only show contour lines, no filled regions
		)
	
	# Increase tick label sizes
	axes = np.array(figure.axes).reshape((len(parameters), len(parameters)))
	for ax in axes.flat:
		ax.tick_params(labelsize=16, length=8, width=2)  # Increased tick size and label size
		ax.grid(False)  # Explicitly turn off grid for each subplot
	
	# Create invisible rectangles for legend
	legend_elements = []
	for label, color in zip(labels, colors):
		rect = plt.Rectangle((0,0), 4, 4, fc=color, alpha=0.5, label=label, ec='none')
		legend_elements.append(rect)
	
	# Add legend with larger size and outside the plot
	axes[0, -1].legend(handles=legend_elements, fontsize=40, bbox_to_anchor=(1.3, 1.0), frameon=False)
	
	# Add quantile text above each diagonal plot
	for i in range(len(parameters)):
		ax = axes[i,i]
		
		for j, (quantiles, color) in enumerate(zip(quantiles_list, colors)):
			median = quantiles[parameters[i]][1]
			lower = quantiles[parameters[i]][0]
			upper = quantiles[parameters[i]][2]
			
			text = f"{latex_label_dict[parameters[i]]} = ${median:.2f}^{{+{upper-median:.2f}}}_{{-{median-lower:.2f}}}$"
			# Both texts positioned above the plot with different vertical offsets
			ax.text(0.5, 1.3 - j*0.15, text,  # Adjusted vertical position
				   horizontalalignment='center',
				   color=color,
				   transform=ax.transAxes,
				   fontsize=12)  # Increased text fontsize
	if filename:
		plt.savefig(filename, bbox_inches='tight', dpi=80)
	plt.close()


def find_ml_phase(injection_parameters, injection_generator, strain_data, psd_H1, psd_L1, farray, ngrid=100):
	phase_grid = np.linspace(0, 2*np.pi, ngrid)
	#likelihood_grid = np.zeros(ngrid)
	d_H1 = strain_data['waveform']['H1']
	d_L1 = strain_data['waveform']['L1']

	max_log_likelihood = -np.inf
	max_s = None 
	max_phase = None
	for i in range(ngrid):
		injection_parameters['phase'] = phase_grid[i]
		s = injection_generator.signal(injection_parameters)
		h_H1 = s['waveform']['H1']
		h_L1 = s['waveform']['L1']
		log_likelihood = -0.5* (bilby.gw.utils.inner_product(d_H1-h_H1, d_H1-h_H1, farray, psd_H1) + bilby.gw.utils.inner_product(d_L1-h_L1, d_L1-h_L1, farray, psd_L1))
		if log_likelihood > max_log_likelihood:
			max_log_likelihood = log_likelihood
			max_phase = phase_grid[i]
			max_s = s

	print(f"max likelihood phase: {max_phase}")
	return max_phase, max_s, max_log_likelihood

def calc_likelihood(args):
	i, phase, injection_parameters, injection_generator, d_H1, d_L1, psd_H1, psd_L1, farray = args
	injection_parameters = deepcopy(injection_parameters)
	injection_parameters['phase'] = phase
	s = injection_generator.signal(injection_parameters)
	h_H1 = s['waveform']['H1']
	h_L1 = s['waveform']['L1']
	log_likelihood = -0.5 * (bilby.gw.utils.inner_product(d_H1-h_H1, d_H1-h_H1, farray, psd_H1) + 
				   bilby.gw.utils.inner_product(d_L1-h_L1, d_L1-h_L1, farray, psd_L1))
	return log_likelihood, s

def find_ml_phase_mp(injection_parameters, injection_generator, strain_data, psd_H1, psd_L1, farray, ncpu=8, ngrid=100):
	phase_grid = np.linspace(0, 2*np.pi, ngrid)
	d_H1 = strain_data['waveform']['H1']
	d_L1 = strain_data['waveform']['L1']
	
	likelihood_grid = np.zeros(ngrid)
	args = [(i, phase_grid[i], injection_parameters, injection_generator, d_H1, d_L1, psd_H1, psd_L1, farray) 
			for i in range(ngrid)]
	
	with ProcessPoolExecutor(max_workers=ncpu) as executor:
		results = list(executor.map(calc_likelihood, args))
	
	likelihood_grid = [r[0] for r in results]
	signals = [r[1] for r in results]
	
	max_idx = np.argmax(likelihood_grid)
	ml_phase = phase_grid[max_idx]
	ml_signal = signals[max_idx]
	
	logger.info(f"max likelihood phase: {ml_phase}")
	return ml_phase, ml_signal

def get_a_valid_sample(result, injection_generator, initial_i = 0):
	#i = initial_i
	i_list = np.random.permutation(len(result.samples))
	for i in i_list:
		inj_para_temp = result.samples.iloc[i].to_dict()
		inj_para_temp['phase'] = 0
		try:
			s = injection_generator.signal(inj_para_temp)
			return inj_para_temp
		except:
			continue
	raise ValueError("No valid sample found")


def remove_signal(strain_data, result, injection_generator, psd_H1, psd_L1):
	injection_parameters = get_a_valid_sample(result, injection_generator)
	#injection_parameters['phase'] = float(np.random.uniform(0, 2*np.pi))
	injection_parameters['phase'] = find_ml_phase(injection_parameters, injection_generator, strain_data, psd_H1, psd_L1, farray) 
	s = injection_generator.signal(injection_parameters)
	h_H1 = s['waveform']['H1']
	h_L1 = s['waveform']['L1']
	strain_data['waveform']['H1'] = strain_data['waveform']['H1'] - h_H1
	strain_data['waveform']['L1'] = strain_data['waveform']['L1'] - h_L1
	return strain_data


def remove_signal_ensemble(strain_data_ensemble_original, strain_data_ensemble, result, injection_generator, psd_H1, psd_L1, farray, Nensemble=100, ncpu=8):

	strain_data_ensemble_subtracted = []
	#subtraction_waveform_list = []
	i=0
	logL_ensemble = []
	while len(strain_data_ensemble_subtracted) < Nensemble:
		strain_data_original = deepcopy(strain_data_ensemble_original[i])
		strain_data = deepcopy(strain_data_ensemble[i])
		#injection_parameters = get_a_valid_sample(result[i], initial_i=len(strain_data_ensemble))
		injection_parameters = get_a_valid_sample(result[i], injection_generator)
		injection_parameters['phase'], s, logL = find_ml_phase(injection_parameters, injection_generator, strain_data, psd_H1, psd_L1, farray) #max_phase, max_s, max_log_likelihood
		#injection_parameters['phase'], s = find_ml_phase_mp(injection_parameters, injection_generator, strain_data_temp, psd_H1, psd_L1, farray, ncpu=ncpu)
		#s = injection_generator.signal(injection_parameters)
		h_H1 = s['waveform']['H1']
		h_L1 = s['waveform']['L1']
		strain_data_original['waveform']['H1'] = strain_data_original['waveform']['H1'] - h_H1
		strain_data_original['waveform']['L1'] = strain_data_original['waveform']['L1'] - h_L1
		strain_data_ensemble_subtracted.append(strain_data_original)
		logL_ensemble.append(logL)
		#subtraction_parameters_list.append(injection_parameters.copy())
		i += 1
	return strain_data_ensemble_subtracted, np.array(logL_ensemble) #, subtraction_parameters_list


def sample_ensemble(sampler, Nsample, strain_data_ensemble, indicies=None):
	result_list = []
	if indicies is None:
		#indicies = range(len(strain_data_ensemble))
		sample_each = Nsample // len(strain_data_ensemble)
		for strain_data_temp in strain_data_ensemble:
			sampler.context = strain_data_temp
			sampler.run_sampler(num_samples=sample_each, batch_size=sample_each)
			result_temp = sampler.to_result()
			result_list.append(result_temp)
		return result_list
	
	# Create a dictionary to group strain data by unique indices
	unique_indices = {}
	for idx, strain_data_temp in zip(indicies, strain_data_ensemble):
		if idx not in unique_indices:
			unique_indices[idx] = []
		unique_indices[idx].append(strain_data_temp)
	
	# Sample for each unique group
	for idx, group in unique_indices.items():
		sample_each = int(Nsample // len(strain_data_ensemble) * len(group))
		sampler.context = group[0]  # All elements in the group are identical
		sampler.run_sampler(num_samples=sample_each, batch_size=sample_each)
		result_temp = sampler.to_result()
		
		# Split the result according to the number of identical elements
		num_samples_per_result = sample_each // len(group)
		for i in range(len(group)):
			split_result = deepcopy(result_temp)
			start_idx = i * num_samples_per_result
			end_idx = start_idx + num_samples_per_result
			split_result.samples = result_temp.samples.iloc[start_idx:end_idx].copy()
			result_list.append(split_result)
	
	return result_list

def combine_result(result_list):
	result_combined = result_list[0]
	for result_temp in result_list[1:]:
		result_combined.samples = pd.concat([result_combined.samples, result_temp.samples])
	return result_combined

def which_source(injection_parameters_1, injection_parameters_2, injection_parameters_temp,
				para_to_check=['ra', 'dec']):
	# Compare all parameters in the list and take mean difference
	diff1 = []
	diff2 = []
	for param in para_to_check:
		param1 = injection_parameters_1[param]
		param2 = injection_parameters_2[param]
		param_temp = injection_parameters_temp[param]
		diff1.append(abs(param1 - param_temp))
		diff2.append(abs(param2 - param_temp))
	
	mean_diff1 = np.mean(diff1)
	mean_diff2 = np.mean(diff2)
	
	if mean_diff1 < mean_diff2:
		logger.info(f"Identified source 1 by mean difference across parameters {para_to_check}")
		return 1
	else:
		logger.info(f"Identified source 2 by mean difference across parameters {para_to_check}")
		return 2

def compare_result(result1, result2, para_to_check=['chirp_mass', 'mass_ratio', 'geocent_time', 'ra', 'dec', 'luminosity_distance', 'theta_jn', 'a_1'], method = "mean"):
	if result1 is None or result2 is None:
		logger.warning(f"One of the result is None, returning 100")
		return 100
	jsd = []
	for param in para_to_check:
		param1 = result1.samples[param].values
		param2 = result2.samples[param].values
		jsd.append(jensen_shannon_divergence([param1,param2],base=2.718))
		logger.info(f"JSD {param}: {jsd[-1]:.5f}")
	# Convert to numpy arrays
	jsd = np.array(jsd)
	if method == "mean":
		mean_jsd = np.mean(jsd)
	elif method == "median":
		mean_jsd = np.median(jsd)
	else:
		raise ValueError(f"Invalid method: {method}")
	return mean_jsd


def which_source_by_result(result_source1only, result_source2only, result_temp, 
						para_to_check=['chirp_mass', 'mass_ratio', 'geocent_time', 'ra', 'dec', 'luminosity_distance', 'theta_jn', 'a_1']):
	
	jsd1 = []
	jsd2 = []
	for param in para_to_check:
		param1 = result_source1only.samples[param].values
		param2 = result_source2only.samples[param].values
		param_temp = result_temp.samples[param].values
		jsd1.append(jensen_shannon_divergence([param1,param_temp],base=2.718))
		jsd2.append(jensen_shannon_divergence([param2,param_temp],base=2.718))
		logger.info(f"JSD {param}: {jsd1[-1]:.5f} {jsd2[-1]:.5f}")
	
	# Convert to numpy arrays
	jsd1 = np.array(jsd1)
	jsd2 = np.array(jsd2)
	
	# If all values in one array are inf/nan, choose the other source
	if np.all(np.isinf(jsd1) | np.isnan(jsd1)):
		logger.info("All JSD values for source 1 are inf/nan, choosing source 2")
		return 2
	if np.all(np.isinf(jsd2) | np.isnan(jsd2)):
		logger.info("All JSD values for source 2 are inf/nan, choosing source 1") 
		return 1
		
	# Otherwise remove inf/nan values for comparison
	mask = ~(np.isinf(jsd1) | np.isinf(jsd2) | np.isnan(jsd1) | np.isnan(jsd2))
	jsd1 = jsd1[mask]
	jsd2 = jsd2[mask]
	mean_jsd1 = np.median(jsd1)
	mean_jsd2 = np.median(jsd2)
	if mean_jsd1 < mean_jsd2:
		logger.info(f"Identified source 1 by mean Jensen-Shannon divergence across parameters {para_to_check}")
		return 1
	else:
		logger.info(f"Identified source 2 by mean Jensen-Shannon divergence across parameters {para_to_check}")
		return 2

def rejection_sample(posterior, weights, logger=None):
	keep = weights > np.random.uniform(0, max(weights), weights.shape)
	fraction_kept = len(weights[keep]) / len(weights)
	if logger is not None:
		logger.info(f"Fraction kept: {fraction_kept}")
	else:
		print(f"Fraction kept: {fraction_kept}")
	return posterior[keep]

def temperature_scaling_resampling(log_likelihoods, T=1):
	log_likelihoods = log_likelihoods - np.max(log_likelihoods)
	likelihoods = np.exp(log_likelihoods)
	temperature_scaled_likelihoods = np.power(likelihoods, 1.0 / T)
	weights = temperature_scaled_likelihoods / np.sum(temperature_scaled_likelihoods)

	print("weights", weights)

	# Resample using systematic resampling
	N = len(log_likelihoods)
	cumulative_sum = np.cumsum(weights)
	positions = (np.arange(N) + np.random.uniform()) / N
	indices = np.searchsorted(cumulative_sum, positions)
	return indices
	

def metropolis_sampling(log_L_old, log_L_new):
	# Calculate acceptance ratios for each sample
	acceptance_ratios = np.exp(log_L_new - log_L_old)
	acceptance_ratios = np.minimum(acceptance_ratios, 1)
	acceptance_ratios = np.maximum(acceptance_ratios, 0)

	# Generate random numbers for acceptance test
	random_numbers = np.random.uniform(0, 1, size=len(log_L_old))

	# Accept/reject each sample
	acceptance = random_numbers < acceptance_ratios

	# Get indices of accepted new samples and kept old samples
	new_indices = np.where(acceptance)[0]
	old_indices = np.where(~acceptance)[0]

	return new_indices, old_indices

'''
def importance_sample_gibbs(result_ensemble, injection_generator, subtraction_parameters_list):
	result_ensemble_is = []
	for i,result in enumerate(result_ensemble):
		posterior = result.samples
		Nsample = len(posterior)
		injection_parameters_subtracted = subtraction_parameters_list[i]

		weights = result.log_likelihoods
		posterior_sampled = rejection_sample(posterior, weights)
		posterior_sampled_list.append(posterior_sampled)

	return
'''

if __name__ == "__main__":
	outdir_base = sys.argv[1] 
	
	Nsample = int(sys.argv[2]) # 5000
	Nensemble = int(sys.argv[3]) # 250
	N_GNPE_iter = int(sys.argv[4]) # 25
	N_Gibbs_iter = int(sys.argv[5]) # 25
	dt = float(sys.argv[6]) / 1000 # 50 / 1000 s
	save_strain_data = int(sys.argv[7])
	load_strain_data_from_path = str(sys.argv[8])
	save_result = int(sys.argv[9])
	ncpu = int(sys.argv[10])
	device = str(sys.argv[11])
	init_temperature = int(sys.argv[12])
	niter_anneal_to_one = int(sys.argv[13])
	resampling_maxrepeat = int(sys.argv[14])
	convergence_jsd_threshold = float(sys.argv[15])
	N_run = int(sys.argv[16])

	START_RUN_ID = 6

	for RUN_ID in range(START_RUN_ID, N_run):
		print(f"Running simulation {RUN_ID}/{N_run}")
		outdir = f"{outdir_base}/results_dt{int(dt*1000)}ms/simulation_{RUN_ID}"
		do_metropolis_sampling = 0
		os.makedirs(outdir, exist_ok=True)
		os.makedirs(f"{outdir}/figures", exist_ok=True)
		os.makedirs(f"{outdir}/results", exist_ok=True)
		
		logger = logging.getLogger('logger')
		logging.basicConfig(
			filename=os.path.join(outdir, 'dingo_os_batchrun.log'),
			level=logging.INFO,
			format='%(asctime)s - %(levelname)s - %(message)s')
		logger.info(f"PID: {os.getpid()}")
		logger.info(f"device: {device}")
		logger.info(f"Nsample: {Nsample}")
		logger.info(f"Nensemble: {Nensemble}")
		logger.info(f"N_GNPE_iter: {N_GNPE_iter}")
		logger.info(f"N_Gibbs_iter: {N_Gibbs_iter}")
		logger.info(f"dt: {dt}s")
		logger.info(f"init_temperature: {init_temperature}")
		logger.info(f"niter_anneal_to_one: {niter_anneal_to_one}")
		logger.info(f"resampling_maxrepeat: {resampling_maxrepeat}")
		logger.info(f"convergence_jsd_threshold: {convergence_jsd_threshold}")

		Nsample_trial = Nsample

		if niter_anneal_to_one > 0 and init_temperature > 0:
			alpha = (1 / init_temperature) ** (1 / (niter_anneal_to_one - 1))
			temperature_list = [init_temperature * alpha**k for k in range(niter_anneal_to_one)]
		else:
			temperature_list = [init_temperature]
		for iii in range(10000):
			temperature_list.append(1)
		temperature = temperature_list[0]

		main_pm = PosteriorModel(
			device=device,
			model_filename="/home/qian.hu/overlap_nf/dingo_models/IMRPhenomXPHM_GW150914_model/main_train_dir/model.pt", 
			load_training_info=False
		)
		#main_pm.model_to_device(device)

		init_pm = PosteriorModel(
			device=device,
			model_filename="/home/qian.hu/overlap_nf/dingo_models/IMRPhenomXPHM_GW150914_model/init_train_dir/model_init.pt",
			load_training_info=False
		)
		#init_pm.model_to_device(device)

		init_sampler = GWSampler(model=init_pm)
		sampler = GWSamplerGNPE(model=main_pm, init_sampler=init_sampler, num_iterations=N_GNPE_iter)
		injection_generator = Injection.from_posterior_model_metadata(main_pm.metadata)
		asd_fname = '/home/qian.hu/overlap_nf/dingo_models/noise/O1/8s/asds_O1_fiducial.hdf5'
		asd_dataset = ASDDataset(file_name=asd_fname)
		injection_generator.asd = {k:v[0] for k,v in asd_dataset.asds.items()}

		f_min = main_pm.metadata['dataset_settings']['domain']['f_min']
		f_max = main_pm.metadata['dataset_settings']['domain']['f_max']
		delta_f = main_pm.metadata['dataset_settings']['domain']['delta_f']
		duration = int(1/delta_f)
		sampling_frequency = 2*f_max

		waveform_arguments = main_pm.metadata['dataset_settings']['waveform_generator']

		ifos = bilby.gw.detector.InterferometerList(["H1","L1"])
		for det in ifos:
			det.duration = duration
			det.sampling_frequency = sampling_frequency
		det_H1 = ifos[0]
		det_L1 = ifos[1]
		farray = det_H1.frequency_array
		
		psd_H1 = bilby.gw.detector.PowerSpectralDensity(frequency_array=det_H1.frequency_array, asd_array=np.float64(injection_generator.asd['H1']))
		psd_L1 = bilby.gw.detector.PowerSpectralDensity(frequency_array=det_L1.frequency_array, asd_array=np.float64(injection_generator.asd['L1']))
		det_H1.power_spectral_density = psd_H1
		det_L1.power_spectral_density = psd_L1

		waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
			duration=duration,
			sampling_frequency=sampling_frequency,
			waveform_arguments=waveform_arguments,
			frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole
		)

		frequency_array = det_H1.frequency_array

		t1 = 0
		t2 = t1 + dt

		snr_source1_net = 0
		snr_source2_net = 0
		snr_threshold = 12
		while snr_source1_net < snr_threshold or snr_source2_net < snr_threshold:
			if load_strain_data_from_path != "0":
				with open(f"{load_strain_data_from_path}/strain_data.pkl", "rb") as f:
					strain_data = pickle.load(f)
				with open(f"{load_strain_data_from_path}/strain_data_source1only.pkl", "rb") as f:
					strain_data_source1only = pickle.load(f)
				with open(f"{load_strain_data_from_path}/strain_data_source2only.pkl", "rb") as f:
					strain_data_source2only = pickle.load(f)
				logger.info(f"Loaded strain data from {load_strain_data_from_path}")
				injection_parameters_1 = strain_data['parameters_1']
				injection_parameters_2 = strain_data['parameters_2']
				snr_source1_net = strain_data['SNR_source1']
				snr_source2_net = strain_data['SNR_source2']
				snr_source1_H1 = strain_data['SNR_source1_H1']
				snr_source1_L1 = strain_data['SNR_source1_L1']
				snr_source2_H1 = strain_data['SNR_source2_H1']
				snr_source2_L1 = strain_data['SNR_source2_L1']

			else:
				injection_parameters_1 = injection_generator.prior.sample()
				injection_parameters_2 = injection_generator.prior.sample()

				injection_parameters_1['geocent_time'] = t1
				injection_parameters_2['geocent_time'] = t2

				h1_dict = waveform_generator.frequency_domain_strain(injection_parameters_1)
				h2_dict = waveform_generator.frequency_domain_strain(injection_parameters_2)

				h1_H1 = det_H1.get_detector_response(h1_dict, injection_parameters_1)
				h1_L1 = det_L1.get_detector_response(h1_dict, injection_parameters_1)
				h2_H1 = det_H1.get_detector_response(h2_dict, injection_parameters_2)
				h2_L1 = det_L1.get_detector_response(h2_dict, injection_parameters_2)

				snr_source1_H1 = bilby.gw.utils.inner_product(h1_H1, h1_H1, det_H1.frequency_array, psd_H1)**0.5
				snr_source1_L1 = bilby.gw.utils.inner_product(h1_L1, h1_L1, det_L1.frequency_array, psd_L1)**0.5
				snr_source1_net = (snr_source1_H1**2 + snr_source1_L1**2)**0.5
				snr_source2_H1 = bilby.gw.utils.inner_product(h2_H1, h2_H1, det_H1.frequency_array, psd_H1)**0.5
				snr_source2_L1 = bilby.gw.utils.inner_product(h2_L1, h2_L1, det_L1.frequency_array, psd_L1)**0.5
				snr_source2_net = (snr_source2_H1**2 + snr_source2_L1**2)**0.5

				injection_parameters_1['ra'] = float(injection_parameters_1['ra'])
				injection_parameters_1['dec'] = float(injection_parameters_1['dec'])
				injection_parameters_2['ra'] = float(injection_parameters_2['ra'])
				injection_parameters_2['dec'] = float(injection_parameters_2['dec'])

				theta1 = {**injection_parameters_1}
				theta2 = {**injection_parameters_2}
				strain_data = injection_generator.injection(theta1)
				strain_data_source1only = deepcopy(strain_data)

				s1_dingo = injection_generator.signal(theta1)
				s2_dingo = injection_generator.signal(theta2)
				h1_H1_dingo = s1_dingo['waveform']['H1']
				h1_L1_dingo = s1_dingo['waveform']['L1']
				h2_H1_dingo = s2_dingo['waveform']['H1']
				h2_L1_dingo = s2_dingo['waveform']['L1']

				strain_data['waveform']['H1'] = strain_data['waveform']['H1'].copy() + h2_H1_dingo
				strain_data['waveform']['L1'] = strain_data['waveform']['L1'].copy() + h2_L1_dingo

				strain_data_source2only = deepcopy(strain_data)
				strain_data_source2only['waveform']['H1'] = strain_data_source2only['waveform']['H1'].copy() - h1_H1_dingo
				strain_data_source2only['waveform']['L1'] = strain_data_source2only['waveform']['L1'].copy() - h1_L1_dingo
				
				strain_data_source2only['parameters'] = injection_parameters_2
				strain_data_source1only['parameters'] = injection_parameters_1

				strain_data['parameters_1'] = injection_parameters_1
				strain_data['parameters_2'] = injection_parameters_2
				strain_data['SNR_source1'] = snr_source1_net
				strain_data['SNR_source2'] = snr_source2_net
				strain_data['SNR_source1_H1'] = snr_source1_H1
				strain_data['SNR_source1_L1'] = snr_source1_L1
				strain_data['SNR_source2_H1'] = snr_source2_H1
				strain_data['SNR_source2_L1'] = snr_source2_L1
		logger.info('Injection info:')
		logger.info('-' * 80)
		logger.info(f"{'Parameters':<25} {'Source 1':<25} {'Source 2':<25}")
		logger.info('-' * 80)
		for param in injection_parameters_1.keys():
			val1 = injection_parameters_1[param]
			val2 = injection_parameters_2[param]
			if isinstance(val1, float):
				logger.info(f"{param:<25} {val1:<25.3f} {val2:<25.3f}")
			else:
				logger.info(f"{param:<25} {str(val1):<25} {str(val2):<25}")
		logger.info('-' * 80)
		logger.info(f"{'SNR H1':<25} {snr_source1_H1:<25.3f} {snr_source2_H1:<25.3f}")
		logger.info(f"{'SNR L1':<25} {snr_source1_L1:<25.3f} {snr_source2_L1:<25.3f}")
		logger.info(f"{'SNR Network':<25} {snr_source1_net:<25.3f} {snr_source2_net:<25.3f}")
		logger.info('-' * 80)

		if save_strain_data:	
			# Save strain data to outdir
			with open(f"{outdir}/strain_data.pkl", "wb") as f:
				pickle.dump(strain_data, f)
			with open(f"{outdir}/strain_data_source1only.pkl", "wb") as f:
				pickle.dump(strain_data_source1only, f)
			with open(f"{outdir}/strain_data_source2only.pkl", "wb") as f:
				pickle.dump(strain_data_source2only, f)
		#Nsample = 5000


		sampler.context = strain_data_source1only
		sampler.run_sampler(num_samples=Nsample, batch_size=min(Nsample, 5000))
		result_source1only = sampler.to_result()

		parameters = ["chirp_mass", "mass_ratio", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl", "theta_jn",  "luminosity_distance", "ra", "dec", "psi", "geocent_time"]
		kwargs = {"legend_font_size": 15, "truth_color": "black"}
		_ = result_source1only.plot_corner(parameters=parameters,
						filename=f"{outdir}/figures/corner_source1only.pdf",
						truths=injection_parameters_1,
						**kwargs)
		plt.close()

		sampler.context = strain_data_source2only
		sampler.run_sampler(num_samples=Nsample, batch_size=min(Nsample, 5000))
		result_source2only = sampler.to_result()
		_ = result_source2only.plot_corner(parameters=parameters,
						filename=f"{outdir}/figures/corner_source2only.pdf",
						truths=injection_parameters_2,
						**kwargs)
		plt.close()

		if save_result:
			logger.info(f"Saving result for source 1- and 2- only")
			with open(f"{outdir}/results/result_source1only.pkl", "wb") as f:
				pickle.dump(result_source1only.samples, f)
			with open(f"{outdir}/results/result_source2only.pkl", "wb") as f:
				pickle.dump(result_source2only.samples, f)

		Ncheck = 1
		niter=0
		old_results = {}
		old_results['source1'] = {}
		old_results['source1']['result'] = None
		old_results['source1']['jsd'] = 916
		old_results['source2'] = {}
		old_results['source2']['result'] = None
		old_results['source2']['jsd'] = 916
		old_results['log_evidence'] = []

		start_time = time.time()
		# Gibbs sampling
		while niter < N_Gibbs_iter:
			logger.info("\n")
			logger.info(f"Simulation {RUN_ID}, Iteration {niter} ")
			if temperature!=0:
				temperature = temperature_list[niter]

			if niter == 0:
				sampler.context = strain_data
				sampler.run_sampler(num_samples=Nsample_trial, batch_size=min(Nsample_trial, 5000))
				result_temp = sampler.to_result()
				result_ensemble = []
				strain_data_ensemble = []
				for nmd in range(Nensemble):
					strain_data_ensemble.append(deepcopy(strain_data))
					result_ensemble.append(result_temp)
				strain_data_ensemble_original = deepcopy(strain_data_ensemble)

				logger.info(f"Removing signal from ensemble {niter} ")
				strain_data_ensemble_previous_iter = deepcopy(strain_data_ensemble)
				strain_data_ensemble, logL_ensemble = remove_signal_ensemble(strain_data_ensemble_original, 
																	strain_data_ensemble_previous_iter, 
																	result_ensemble, 
																	injection_generator, 
																	psd_H1, 
																	psd_L1, 
																	farray, 
																	Nensemble=Nensemble)
				log_evidence = scipy.special.logsumexp(logL_ensemble) - np.log(Nensemble)
			else:
				logger.info(f"Sampling ensemble {niter}")
				if niter > 1:
					result_ensemble_previous_2iter = deepcopy(result_ensemble_previous_iter)
					strain_data_ensemble_previous_2iter = deepcopy(strain_data_ensemble_previous_iter)
					logL_ensemble_previous_2iter = deepcopy(logL_ensemble_previous_iter)
				
				result_ensemble_previous_iter = deepcopy(result_ensemble)
				strain_data_ensemble_previous_iter = deepcopy(strain_data_ensemble)
				logL_ensemble_previous_iter = deepcopy(logL_ensemble)


				result_ensemble = sample_ensemble(sampler, Nsample_trial, strain_data_ensemble, indicies=indicies)
				result_temp = combine_result(result_ensemble)

				logger.info(f"Removing signal from ensemble {niter} and calculating evidence")
				
				strain_data_ensemble, logL_ensemble = remove_signal_ensemble(strain_data_ensemble_original, 
																	strain_data_ensemble_previous_iter, 
																	result_ensemble, 
																	injection_generator, 
																	psd_H1, 
																	psd_L1, 
																	farray, 
																	Nensemble=Nensemble)
				log_evidence = scipy.special.logsumexp(logL_ensemble) - np.log(Nensemble)
				
				if not do_metropolis_sampling:
					logger.info(f"Using greey algorithm to find a better evidence for iteration {niter}. ")
					nrepeat = 1
					while (log_evidence < old_results['log_evidence'][-1]):
						#if nrepeat == 1:
						#	strain_data_ensemble_backup = deepcopy(strain_data_ensemble_previous_iter)
						#	result_ensemble_backup = deepcopy(result_ensemble_previous_iter)
						#	logL_ensemble_backup = deepcopy(logL_ensemble_previous_iter)

						if nrepeat > resampling_maxrepeat:
							#logger.warning(f"Can't find a better log evidence, going with current value")
							#logger.warning(f"Can't find a better log evidence! Setting niter = {N_Gibbs_iter} and Gibbs sampling will not enter the next iteration")
							#niter = N_Gibbs_iter

							logger.warning(f"Can't find a better log evidence! Start Metropolis sampling")
							
							do_metropolis_sampling = 1
							temperature=0
							#strain_data_ensemble = strain_data_ensemble_backup
							#result_ensemble = result_ensemble_backup
							#logL_ensemble = logL_ensemble_backup
							#log_evidence = scipy.special.logsumexp(logL_ensemble) - np.log(Nensemble)
							break

						logger.info(f"Log evidence for iteration {niter} = {log_evidence:.5f} is lower than the previous iteration, removing signal again ")
						strain_data_ensemble, logL_ensemble = remove_signal_ensemble(strain_data_ensemble_original, 
																		strain_data_ensemble_previous_iter, 
																		result_ensemble, 
																		injection_generator, 
																		psd_H1, 
																		psd_L1, 
																		farray, 
																		Nensemble=Nensemble)
						log_evidence = scipy.special.logsumexp(logL_ensemble) - np.log(Nensemble)
						nrepeat += 1



				if do_metropolis_sampling:
					if niter > 1:
						logger.info(f"Metropolis sampling for iteration {niter}")
						new_indices, old_indices = metropolis_sampling(logL_ensemble_previous_iter, logL_ensemble)
						logger.info(f"Metropolis sampling chooses {len(new_indices)} new samples and {len(old_indices)} old samples")

						for i_old in old_indices:
							strain_data_ensemble[i_old] = strain_data_ensemble_previous_2iter[i_old]
							result_ensemble[i_old] = result_ensemble_previous_2iter[i_old]
							logL_ensemble[i_old] = logL_ensemble_previous_iter[i_old] # be careful! 

						result_temp = combine_result(result_ensemble)
						log_evidence = scipy.special.logsumexp(logL_ensemble) - np.log(Nensemble)
					else:
						raise ValueError(f"Metropolis sampling cannot be applied for iteration {niter} < 2. ")
						
				
			if temperature==0:
				logger.info(f"Not applying weighted resampling. ")
				indicies = None
			else:
				temperature = temperature_list[niter]
				logger.info(f"Resampling for iter {niter} with T={temperature}")
				indicies = temperature_scaling_resampling(logL_ensemble, temperature)
				N_unique_indicies = len(np.unique(indicies))
				logger.info(f"Resampling chooses {N_unique_indicies} unique samples out of {Nensemble} ensembles")
				strain_data_ensemble = [strain_data_ensemble[i] for i in indicies]
				result_ensemble = [result_ensemble[i] for i in indicies]
				if niter > 0:
					logger.info('combining result')
					result_temp_resampled = combine_result(result_ensemble)
					logger.info('combining result done')

			old_results['log_evidence'].append(log_evidence)
			logger.info(f"Log evidence for iteration {niter}: {log_evidence:.5f} ")

			logger.info("Making checkpoint plot for iteration ", niter)
			source = which_source_by_result(result_source1only, result_source2only, result_temp,
									para_to_check=['chirp_mass', 'geocent_time', 'dec', 'luminosity_distance'])
			result_baseline = result_source1only if source == 1 else result_source2only
			injection_parameters_to_plot = injection_parameters_1.copy() if source == 1 else injection_parameters_2.copy()
			injection_parameters_to_plot.pop('phase', None)
			
			#if indicies is not None and niter > 0:
			if 0:
				samples_to_plot = [result_temp.samples, result_baseline.samples, result_temp_resampled.samples]
				labels_to_plot = ['Overlapping', 'Single', 'Overlapping-resampled']
				colors = ['c', 'sandybrown', 'lime']
			else:
				samples_to_plot = [result_baseline.samples, result_temp.samples]
				labels_to_plot = ['Single', 'Overlapping']
				#colors = ['c', 'sandybrown']
				colors = ['tomato', 'forestgreen']
			#_=plot_corner_multi(samples=samples_to_plot,
			#	labels=labels_to_plot,
			#	filename=f"{outdir}/figures/corner_comparison_source{int(source)}_iter{int(niter)}.pdf")
			#plt.close()
			make_corner_plot(samples_to_plot, injection_parameters_to_plot, labels_to_plot, colors, filename=f"{outdir}/figures/corner_comparison_source{int(source)}_iter{int(niter)}.pdf")

			if save_result:
				logger.info(f"Saving result for iteration {niter} source {source}")
				with open(f"{outdir}/results/result_source{int(source)}_iter{int(niter)}.pkl", "wb") as f:
					pickle.dump(result_temp.samples, f)
				if indicies is not None and niter > 0:
					with open(f"{outdir}/results/result_source{int(source)}_iter{int(niter)}_resampled.pkl", "wb") as f:
						pickle.dump(result_temp_resampled.samples, f)

			# check convergence
			if niter > 1:
				jsd = compare_result(old_results[f'source{source}']['result'], result_temp, para_to_check=['chirp_mass', 'geocent_time', 'dec', 'luminosity_distance'])
				old_results[f'source{source}']['jsd'] = jsd
				logger.info(f"JSD {source} from the previous iteration: {jsd:.5f}")


			logger.info(f"Compared with previous iteration: JSD source1: {old_results['source1']['jsd']:.5f}, JSD source2: {old_results['source2']['jsd']:.5f}")

			if old_results[f'source1']['jsd'] < convergence_jsd_threshold and old_results[f'source2']['jsd'] < convergence_jsd_threshold:
				logger.info(f"Convergence reached the threshold {convergence_jsd_threshold}")
				if abs(temperature - 0) < 1e-3 and do_metropolis_sampling:
					logger.info(f"Resampling has already stopped, stopping Gibbs sampling")
					break
				else:
					logger.info(f"However temperature has not reached 1, continuing Gibbs sampling")
			else:
				logger.info(f"Convergence not reached, continuing Gibbs sampling")
			
			old_results[f'source{source}']['result'] = deepcopy(result_temp)
			niter += 1

			if niter == N_Gibbs_iter:
				logger.info(f"Gibbs sampling reached the maximum number of iterations {N_Gibbs_iter}. Exiting. ")


		timecost = time.time() - start_time
		logger.info("Analysis finished. Gibbs sampling time cost: %s seconds\n\n\n----------------------------------\n\n\n", timecost)
		np.savetxt(f"{outdir}/time_cost.txt", [niter, timecost])