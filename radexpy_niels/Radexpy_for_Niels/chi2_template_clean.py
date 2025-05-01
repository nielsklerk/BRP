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

# Molecule to fit
mol = 'CO'

# Read in your data however you want
file = f'FullSpectrum_CS_V1094Sco.p'
data = pickle.load(open(file, 'rb'))

o_w = data['Wavelength'] # Wavelength array
o_f = data['Flux'] # Flux array
o_f_contsub = data['CSFlux'] # Continuum-subtracted flux array

# clip to useful range where we want to compute/plot our model
clip_min, clip_max = 4.9, 7.0  # micron

clip_cnd = ((o_w >= clip_min) & (o_w <= clip_max))
o_w = o_w[clip_cnd]
o_f_contsub = o_f_contsub[clip_cnd]

for molecule in ['H2O']:
    mol_data = pickle.load(open(f'{molecule}_best_fit.p', 'rb'))
    o_f_contsub -= mol_data['m_f']*mol_data['Rdisk_best']**2


# convert to masked array type 
o_w_mask = ma.array(o_w, mask=False)
o_f_contsub_mask = ma.array(o_f_contsub, mask=False)

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
noise_est = 1e-3 # You can measure the noise by taking a line free region and taking the std of that, but 1 mJy is likely an okay estimate to start with
sigma = noise_est * np.ones_like(o_f_contsub)

# It's probably a good idea to specify small windows in which you fit your models, here are some windows that I had set for my CX Tau paper which can give you some indication of what works
# These cover the full MIRI wavelength range, but we're only interested in the small wavelengths, so feel free to ignore the rest
# True = fit ONLY these windows 
# False = do NOT fit these windows
fit_regions = True

if mol == "CO2":
    color = "limegreen"
    regions = [[13.87, 13.91], # CO2 hot band (left)
           [14.5, 14.6],
           #[14.92, 14.99], # CO2 Q branch
           [15.43, 15.53],
           [16.18, 16.22],  # CO2 hot band (right) # Do not fit due to artifact
           [16.72, 16.78]
           ]

elif mol == "HCN":
    color = "orange"
    regions = [#[13.75, 13.86],
            [13.91, 14.1], # As in Sierra's paper
            [14.29, 14.32]]          

elif mol == "C2H2":
    color = "yellow"
    regions = [[13.62, 13.73]  # Main feature
           #[13.87, 13.91]
           ]      #CO2 left hot band    

elif mol == "H2O":
    color = "dodgerblue"
    regions = [#[5.23, 5.25],
            [5.27, 5.283],
            [5.318, 5.327],
            #[5.47, 5.492],
            [5.607, 5.63],
            #[5.64, 5.696],
           [5.76, 5.79],
           [5.962, 5.978],
           [6.0, 6.2],
           [6.6, 6.9],
           [7.2, 7.4],
           [7.55, 7.7],
           [7.75, 8.],
           [8.2, 8.4],
           [11.2, 11.35],
           [12.35, 12.6],
           [13.19, 13.5],
           [14.46, 14.56],
           [16.09, 16.13],
           [16.625, 16.69],
           [17.07, 17.25],
           [17.314, 17.417],
           [17.48, 17.52],
           [17.55, 17.58], 
           [19.4, 19.62],
           [21.09, 21.44],
           [22.51, 22.56],
           [23.1, 23.3],
           [23.44, 23.71],
           [23.8, 23.96]
           ]

elif mol == "OH":
    color = "magenta"
    regions = [[13.689, 13.707], 
           [14.036, 14.09],
           [14.61, 14.68], 
           [15.26, 15.325],
           [15.89, 16.06],
           [16.78, 16.89],
           [17.71, 17.83],
           [18.77, 18.90],
           [19.98, 20.14],
           [21.39, 21.58],
           [23.05, 23.27]
           ]
           
elif mol == "13CO2":
    color = "blueviolet"
    regions = [[15.39, 15.43]  # Main feature
           ]         

           
elif mol == "13C2H2":
    regions = [[13.70, 13.75]  # Main feature
           ]      

elif mol == "CO":
    color = "red"
    fit_regions = False
    regions = [[5.12, 5.137]  #Mask H line
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

# assumed source distance and velocity
# Change these to the proper values for each source
# don't worry if you cannot find the velocity, since that likely will not matter too much
distance = 150  # pc
v_obs = -8.4  # km/s 

# Instrumental R value assumed for model
# MIRI has a higher spectral resolving power at short wavelengths
if clip_max <= 10.:
    R = 3500
elif clip_max > 10. and clip_max < 18.:
    R = 2500
else:
    R = 1500


# Set up your model grid parameters here, note that you may need to use a higher T for CO
if mol == '13CO2':
    Tgas_list = np.linspace(50, 500, 40)
elif mol == 'OH':
    Tgas_list = np.linspace(500, 1500, 40)
else:
    Tgas_list = np.linspace(100, 1900, 40)
    
N_list = np.logspace(15, 24, 40)

# Rdisk grid for brute forcing
#Rdisk_list = np.linspace(0.01, 3.0, 1000)

# Rdisk intial guess for minimization
Rdisk_guess = 0.5


# ----------------------------------------------------------------------------#
chi2_map = np.zeros((len(N_list), len(Tgas_list)))
best_Rdisk_map = np.zeros((len(N_list), len(Tgas_list)))
# loop over models, calculate chi2. Note: this is embarassingly parallel! Possible
# to accelerate this significantly on multi-core machines.
for i, N in enumerate(N_list):
    for j, Tgas in enumerate(Tgas_list):

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
        chi2_map[i, j] = res.fun
        best_Rdisk_map[i, j] = np.abs(res.x)  # Rdisk must be >= 0

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
#plt.ylim((plot_ylim))
plt.ylabel('Flux Density (Jy)')
plt.xlabel(r'$\lambda$ (um)')
plt.title(mol)
plt.legend()
#plt.savefig('{}_best_fit_{}_Tgas={:.0f}_N={:.1E}_Rdisk={:.1e}.png'.format(mol, source, Tgas_best, N_best, Rdisk_best))
plt.show()
plt.close()

# plot normalized chi2 map and minimum chi2
map_extent = [Tgas_list[0], Tgas_list[-1],
              np.log10(N_list[0]), np.log10(N_list[-1])]
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
#plt.savefig('chi2_map_{}_{}.png'.format(mol, source))
plt.show()
plt.close()

