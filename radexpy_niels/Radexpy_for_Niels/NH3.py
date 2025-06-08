# template script for chi2 fitting molecular emission using radexpy v6

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

sys.path.append('radex-version-IR_clean_v6/')
import radexpy.run_plot as radexpy
import pickle
from spectres.spectral_resampling_numba import spectres_numba as spectres
import numpy.ma as ma
import matplotlib.ticker as ticker
from scipy.optimize import minimize
from scipy.stats import chi2
from multiprocessing import Pool
import os

def run_model(args):
    i, N, j, Tgas, mol = args
    # Calculate LTE model with/without opacity overlap
    if opacity_overlap:
        wlth, flux = radexpy.disk_convo_overlap(species=mol, T=Tgas, N=N, DV=4.71, R=R, Rdisk=1.0,
                                                distance=distance, wlthmin=clip_min, wlthmax=clip_max)
    else:
        wlth, flux = radexpy.disk_convo(species=mol, T=Tgas, N=N, DV=4.71, R=R, Rdisk=1.0,
                                        distance=distance, wlthmin=clip_min, wlthmax=clip_max)

    # velocity shift model
    c = 2.99792458e5  # km /s
    wlth *= (1.0 + v_obs / c)

    # resample on observed wl grid
    m_w = o_w
    m_f = spectres(m_w, wlth, flux, verbose=False, fill=0.0)

    # mask model regions in same manner as observed data
    m_w_mask = ma.array(m_w, mask=False)
    m_f_mask = ma.array(m_f, mask=False)
    if fit_regions:
        m_f_mask.mask = True
        m_w_mask.mask = True
        for reg in regions:
            m_f_mask.mask[(m_w >= reg[0]) & (m_w <= reg[1])] = False
            m_w_mask.mask[(m_w >= reg[0]) & (m_w <= reg[1])] = False
    else:
        for reg in regions:
            m_f_mask.mask[(m_w >= reg[0]) & (m_w <= reg[1])] = True
            m_w_mask.mask[(m_w >= reg[0]) & (m_w <= reg[1])] = True

    # calculate chi2 between model, obs
    # find best fit rdisk using scipy minimization
    def chi2_Rdisk(Rdisk):
        m_f_scaled = m_f_mask * Rdisk ** 2
        chi2 = np.sum(((o_f_contsub_mask - m_f_scaled) / sigma) ** 2)
        return chi2

    res = minimize(chi2_Rdisk, np.array(Rdisk_guess, ), method='Nelder-Mead', tol=1e-6)
    # chi2_map[i, j] = res.fun
    # best_Rdisk_map[i, j] = np.abs(res.x)  # Rdisk must be >= 0

    return i, j, res.fun, np.abs(res.x)


num_cores = os.cpu_count() - 1

artefact_regions = [[5.0091,5.01071], [5.018,5.019], [5.112,5.15],
[5.2157,5.2184], [5.2267,5.2290], [5.2441,5.2471],
[5.2947,5.2974], [5.3742,5.3777], [5.3836,5.3877],
[5.4181,5.4210], [5.5644,5.5674], [5.5925,5.5966],
[5.8252,5.8267], [5.8669,5.8689], [5.9,5.916], [5.9282,5.9314],
[5.9691,5.9728], [6.0357,6.0394], [6.0430,6.0462],
[6.1012,6.1044], [6.1311,6.1421], [6.26,6.31], [6.3740,6.3757],
[6.3783,6.3810], [18.8055,18.8145], [19.004,19.012],
[21.974,21.985], [25.69824,25.71313]]

subtract = True
model_index = 2

# Molecule to fit
mol = 'NH3'
color = 'cyan'

# Read in your data however you want
source = ['GWLup', 'Sz98', 'V1094Sco'][model_index]
file = f'FullSpectrum_CS_{source}.p'
data = pickle.load(open(file, 'rb'))

# assumed source distance and velocity
# Change these to the proper values for each source
distance = [155.20, 156.27, 152.44][model_index] # pc
v_obs = [-3.3, -1.4, 2.2][model_index]  # km/s

o_w = data['Wavelength']  # Wavelength array
o_f = data['Flux']  # Flux array
o_f_contsub = data['CSFlux']  # Continuum-subtracted flux array

noise_est = np.std(o_f_contsub[(o_w>15.90)&(o_w<15.94)])  # You can measure the noise by taking a line free region and taking the std of that, but 1 mJy is likely an okay estimate to start with

# clip to useful range where we want to compute/plot our model
clip_min, clip_max = 4.9, 6.5  # micron

clip_cnd = ((o_w >= clip_min) & (o_w <= clip_max))
o_w = o_w[clip_cnd]
o_f_contsub = o_f_contsub[clip_cnd]

for molecule in ['H2O', 'CO']:
    if molecule == mol:
        continue
    file = f'{source}4_9_6_3/{molecule}_best_fit.p'
    mol_data = pickle.load(open(file, 'rb'))
    m_f = mol_data['m_f']
    m_f = np.nan_to_num(m_f, nan=0)
    # print(
    #     f'{molecule}: N= {mol_data['N_best']:.1e} cm^-2 | T= {mol_data["Tgas_best"]:.0f} K | Rdisk= {mol_data["Rdisk_best"]:.2f} au')
    o_f_contsub -= mol_data['m_f'] * mol_data['Rdisk_best'] ** 2


mask = np.ones_like(o_w, dtype=bool)
for start, end in artefact_regions:
    mask &= ~((o_w >= start) & (o_w <= end))

o_w[~mask] = np.nan
o_f_contsub[~mask] = np.nan


# convert to masked array type
o_w_mask = ma.array(o_w, mask=False)
o_f_contsub_mask = ma.array(o_f_contsub, mask=False)
sigma = noise_est * np.ones_like(o_f_contsub)
# opacity overlap correction? Important for optically thick emission from molecules with a Q branch but much slower
# Likely not important for CO or H2O, can set to False!
opacity_overlap = False

# save best-fit?
save_best_fit = True
if opacity_overlap:
    best_file = f'{mol}_best_fit_overlap.p'
else:
    best_file = f'{mol}_best_fit.p'

#  noise in Jy as array with same length as o_f_contsub
# It's probably a good idea to specify small windows in which you fit your models, here are some windows that I had set for my CX Tau paper which can give you some indication of what works
# These cover the full MIRI wavelength range, but we're only interested in the small wavelengths, so feel free to ignore the rest
# True = fit ONLY these windows
# False = do NOT fit these windows
fit_regions = True
regions = [[6, 6.5]  # Mask H line
           ]

if fit_regions:
    o_f_contsub_mask.mask = True
    o_w_mask.mask = True
    for reg in regions:
        o_f_contsub_mask.mask[(o_w >= reg[0]) & (o_w <= reg[1])] = False
        o_w_mask.mask[(o_w >= reg[0]) & (o_w <= reg[1])] = False
else:
    for reg in regions:
        o_f_contsub_mask.mask[(o_w >= reg[0]) & (o_w <= reg[1])] = True
        o_w_mask.mask[(o_w >= reg[0]) & (o_w <= reg[1])] = True


# Instrumental R value assumed for model
# MIRI has a higher spectral resolving power at short wavelengths
if clip_max <= 10.:
    R = 3500
elif clip_max > 10. and clip_max < 18.:
    R = 2500
else:
    R = 1500

Tgas_list = np.linspace(700, 2500, 80)
N_list = np.logspace(14, 22, 80)

# Rdisk_list = np.linspace(0.01, 3.0, 1000)

# Rdisk intial guess for minimization
Rdisk_guess = 0.06

if __name__ == "__main__":
    # ----------------------------------------------------------------------------#
    chi2_map = np.zeros((len(N_list), len(Tgas_list)))
    best_Rdisk_map = np.zeros((len(N_list), len(Tgas_list)))
    # loop over models, calculate chi2. Note: this is embarassingly parallel! Possible
    # to accelerate this significantly on multi-core machines.

    tasks = [(i, N, j, Tgas, mol) for i, N in enumerate(N_list) for j, Tgas in enumerate(Tgas_list)]

    with Pool(num_cores) as pool:
        results = pool.map(run_model, tasks)

    for i, j, chi2, Rdisk in results:
        chi2_map[i, j] = chi2
        best_Rdisk_map[i, j] = Rdisk

    # Plot the model fit at the chi2 minimum
    best_i, best_j = np.unravel_index(np.argmin(chi2_map, axis=None), chi2_map.shape)
    Tgas_best = Tgas_list[best_j]  # note order
    N_best = N_list[best_i]
    Rdisk_best = best_Rdisk_map[best_i, best_j]

    # Calculate best fit LTE model
    if opacity_overlap:
        wlth, flux = radexpy.disk_convo_overlap(species=mol, T=Tgas_best, N=N_best, DV=4.71, R=R, Rdisk=1.0,
                                                distance=distance, wlthmin=clip_min, wlthmax=clip_max)
    else:
        wlth, flux = radexpy.disk_convo(species=mol, T=Tgas_best, N=N_best, DV=4.71, R=R, Rdisk=1.0,
                                        distance=distance, wlthmin=clip_min, wlthmax=clip_max)

    # velocity shift, absorption correction, regrid
    c = 2.99792458e5  # km /s
    wlth *= (1.0 + v_obs / c)

    # resample on observed wl grid
    m_w = o_w
    m_f = spectres(m_w, wlth, flux, verbose=False, fill=0.0)

    # save residuals and best fit model
    if save_best_fit:
        r_f = o_f_contsub - m_f * Rdisk_best ** 2
        best_fit = {
            'o_w': o_w,
            'm_f': m_f,
            'r_f': r_f,
            'Tgas_best': Tgas_best,
            'N_best': N_best,
            'Rdisk_best': Rdisk_best
        }
        pickle.dump(best_fit, open(best_file, 'wb'))

    # plot best fit model, data
    plt.figure(figsize=(12, 4))
    plt.step(o_w, o_f_contsub, label='obs data', where='mid', color="black")
    plt.step(m_w, m_f * Rdisk_best ** 2, where='mid',
             label='Tgas={:.0f}, N={:.1E}, Rdisk={:.3f}'.format(Tgas_best, N_best, Rdisk_best), color=color)

    # indicate regions used in fit
    plt.plot(o_w_mask, np.ones_like(o_w_mask) * -1.0e-3, color=color)
    plt.hlines(0., clip_min, clip_max, color="grey")

    # plot decorations
    plt.xlim((clip_min, clip_max))
    # plt.ylim((plot_ylim))
    plt.ylabel('Flux Density (Jy)')
    plt.xlabel(r'$\lambda$ (um)')
    plt.title(mol)
    plt.legend()
    # plt.savefig('{}_best_fit_{}_Tgas={:.0f}_N={:.1E}_Rdisk={:.1e}.png'.format(mol, source, Tgas_best, N_best, Rdisk_best))
    plt.show()
    plt.close()

    # plot normalized chi2 map and minimum chi2
    map_extent = [Tgas_list[0], Tgas_list[-1],
                  np.log10(N_list[0]), np.log10(N_list[-1])]
    N_list = np.log10(N_list)
    chi2_min = np.min(chi2_map)
    chi2_norm = chi2_min / chi2_map
    m = plt.imshow(chi2_norm, origin='lower', extent=map_extent, aspect='auto', cmap='inferno')
    # overlay best fit Tgas, N
    plt.scatter(Tgas_best, np.log10(N_best), marker='x', color='b')
    plt.colorbar(m, label=r'$\chi^2_{min}/\chi^2$')

    # overlay best fit Rdisk contours
    Rdisk_levels = np.logspace(-2, 2, 9)
    Rdisk_contours = plt.contour(best_Rdisk_map, levels=Rdisk_levels, origin='lower',
                                 extent=map_extent, colors='white', zorder=2)
    fmt = ticker.LogFormatterMathtext()
    fmt.create_dummy_axis()
    plt.clabel(Rdisk_contours, Rdisk_contours.levels, fmt=fmt)

    # overlay confidence interval 68/95/99.99 contours
    dof = 2
    coeff_all = radexpy.chi2_coef(dof)
    sigma_contour_to_plot = [1, 2, 3]
    contour_levels = []
    for sigma_contour in sigma_contour_to_plot:
        coef = coeff_all[sigma_contour - 1]
        sigma_value = np.nanmin(chi2_map) + coef  # chi2 map used here
        contour_levels.append(sigma_value)

    plt.contour(chi2_map, levels=contour_levels, origin='lower', extent=map_extent, colors='k', zorder=10)

    # plot decorations
    plt.xlabel(r'T$_K$ in K')
    plt.ylabel(r'log10(N({})) in cm$^{{-2}}$'.format(mol))
    plt.title(mol)
    plt.savefig('chi2_map_{}_{}.pdf'.format(mol, source), bbox_inches='tight')
    plt.show()
    plt.close()

    T_grid, N_grid = np.meshgrid(Tgas_list, N_list, indexing='ij')
    log_prior = np.full_like(T_grid, -np.inf, dtype=np.float64)
    log_prior[(T_grid >= 750)] = 0.0

    # Log posterior
    log_posterior = -0.5 * chi2_map + log_prior

    # Optional: exponentiate for posterior probability grid
    log_posterior_flat = log_posterior.ravel()
    posterior_flat = np.exp(log_posterior_flat - np.max(log_posterior_flat))
    posterior_flat /= posterior_flat.sum()

    # Sample indices from the flattened array
    N_samples = 100000
    sampled_indices = np.random.choice(posterior_flat.size, size=N_samples, p=posterior_flat)
    i_sampled, j_sampled = np.unravel_index(sampled_indices, log_posterior.shape)
    # Convert to physical N and T values
    T_sampled_values = np.unique(Tgas_list)[i_sampled]
    N_sampled_values = np.unique(N_list)[j_sampled]

    # Compute upper limits (e.g. 95% for N)
    N_95_upper = np.percentile(N_sampled_values, 95)
    print(f"95% upper limit on N: {N_95_upper:.2f}")
    nx = len(np.unique(Tgas_list))
    ny = len(np.unique(N_list))

    map_extent = [Tgas_list[0], Tgas_list[-1],
                  N_list[0], N_list[-1]]

    T_bins = np.linspace(min(T_sampled_values), max(T_sampled_values), len(np.unique(T_sampled_values)) + 1)
    N_bins = np.linspace(min(N_sampled_values), max(N_sampled_values), len(np.unique(N_sampled_values)) + 1)

    # Compute 2D histogram
    H, T_edges, N_edges = np.histogram2d(T_sampled_values, N_sampled_values, bins=[T_bins, N_bins], density=True)

    # Plot with imshow
    plt.pcolormesh(T_edges, N_edges, H.T, cmap='inferno', shading='auto')
    plt.axhline(N_95_upper, c='white')
    plt.text(1200, N_95_upper + 0.3, f'Upper Limit = {N_95_upper:.1f}', fontsize=14, color='white', ha='center',
             va='bottom')
    plt.xlabel(r'T$_K$ in K')
    plt.ylabel(r'log10(N({})) in cm$^{{-2}}$'.format(mol))
    plt.colorbar(label="Sample Density")

    plt.tight_layout()
    plt.savefig('upper_{}_{}.pdf'.format(mol, source), bbox_inches='tight')
    plt.show()
