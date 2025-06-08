from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import prodimopy.read_slab as rs
import prodimopy.plot_slab as ps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.signal import correlate, fftconvolve, correlate
from scipy.stats import norm, kendalltau, spearmanr, pearsonr
from scipy.optimize import curve_fit
from tqdm import tqdm
from multiprocessing import Pool
from scipy.constants import h as planck_h
from scipy.constants import k as boltzmann_k
from scipy.constants import c
from scipy.constants import astronomical_unit as au
from scipy.constants import parsec as pc
from spectres import spectres
import matplotlib.ticker as ticker


def check_chi(index):
    Rdisk = np.linspace(0.5, 1, 10)
    slab_model = rs.read_slab(f'NO/NO_{index}.fits.gz', verbose=True)
    slab_model.convolve(R=3000, overlap=True, NLTE=False, vr=vr, verbose=False)
    min_chi = rs.red_chi2_slab(slab_model, spectra, distance=distance, Rdisk=Rdisk, mask=mask, overlap=True, noise_level=noise_est)
    return min(min_chi), Rdisk[np.argmin(min_chi)]


def mask_regions(wavelength, regions):
    mask = np.ones_like(wavelength, dtype=bool)
    for region in regions:
        mask[np.logical_and(wavelength > region[0], wavelength < region[1])] = False
    return mask

model_index = 2

Source = ['GWLup', 'Sz98', 'V1094Sco'][model_index]

file = f'FullSpectrum_CS_{Source}.p'
data = pickle.load(open(file, 'rb'))
wavelength = data['Wavelength']
flux_cont_sub = data['CSFlux']

noise_est = np.std(flux_cont_sub[(wavelength>15.90)&(wavelength<15.94)])
print(noise_est)
clip_min, clip_max = 4.9, 6.5  # micron

clip_cnd = ((wavelength >= clip_min) & (wavelength <= clip_max))
total = flux_cont_sub[clip_cnd].copy()
for mol in ['H2O', 'CO']:
    file = f'radexpy_niels/Radexpy_for_Niels/{Source}4_9_6_3/{mol}_best_fit.p'
    data = pickle.load(open(file, 'rb'))
    o_w = data['o_w']
    # o_w = np.nan_to_num(o_w, nan=0)
    m_f = data['m_f']
    m_f = np.nan_to_num(m_f, nan=0)
    # plt.plot(o_w, mask_regions(o_w, 1000*m_f*data['Rdisk_best']**2, artefact_regions), label=mol)
    total -= m_f * data['Rdisk_best'] ** 2
spectra = np.column_stack((wavelength[clip_cnd], total))

artefact_regions = [[5.0091, 5.01071], [5.018, 5.019], [5.112, 5.15],
                    [5.2157, 5.2184], [5.2267, 5.2290], [5.2441, 5.2471],
                    [5.2947, 5.2974], [5.3742, 5.3777], [5.3836, 5.3877],
                    [5.4181, 5.4210], [5.5644, 5.5674], [5.5925, 5.5966],
                    [5.8252, 5.8267], [5.8669, 5.8689], [5.9, 5.916], [5.9282, 5.9314],
                    [5.9691, 5.9728], [6.0357, 6.0394], [6.0430, 6.0462],
                    [6.1012, 6.1044], [6.1311, 6.1421], [6.26, 6.31], [6.3740, 6.3757],
                    [6.3783, 6.3810], [18.8055, 18.8145], [19.004, 19.012],
                    [21.974, 21.985], [25.69824, 25.71313]]

mask = mask_regions(wavelength[clip_cnd], artefact_regions)
# mask = np.ones_like(wavelength[clip_cnd], dtype=bool)
data = np.loadtxt('NT_index.dat')
index_list = data[:, 0]
T_list = data[:, 1]
N_list = data[:, 2]


distance = [155.20, 156.27, 152.44][model_index]
vr = [-3300, -1400, 2200][model_index]

# test_slab = rs.read_slab(f'NO/NO_{int(index_list[(T_list==800)&(N_list==16)]):04d}.fits.gz', verbose=True)
# test_slab.convolve(R=3000, overlap=True, NLTE=False, vr=vr)
# for rdisk in np.linspace(0.5, 1, 10):
#     area=np.pi*(rdisk*au/distance/pc)**2
#     modelSpec=spectres(spectra[:,0],c/test_slab.convOverlapFreq[::-1]*1e-3,test_slab.convOverlapLTE[::-1]*1e23,verbose=False,fill=0.0)
#     norm_modelSpec = np.trapezoid(modelSpec, wavelength[clip_cnd])
#     fig,ax=plt.subplots(figsize=(12,4))
#     ax.step(wavelength[clip_cnd], modelSpec*area*1000)
#     ax.step(wavelength[clip_cnd], total*1000)
#     ax.set_xlim([4.9, 6.5])
#     ax.set_ylim([-5, 15])
#     plt.show()


if __name__ == '__main__':
    chis = np.zeros_like(index_list)
    Rdisks = np.zeros_like(index_list)
    tasks = [f'{int(k):04d}' for k in index_list]
    with Pool(11) as pool:
        results = pool.map(check_chi, tasks)
    for i, (chi, r)  in enumerate(results):
        chis[i] = chi
        Rdisks[i] = r
    log_likelihood = -0.5 * chis
    log_prior = np.full_like(T_list, -np.inf)
    log_prior[(T_list >= 400)] = 0

    # Compute unnormalized log-posterior
    log_posterior = log_likelihood + log_prior
    posterior = np.exp(log_posterior - np.max(log_posterior))  # prevent underflow
    posterior /= np.sum(posterior)
    posterior_grid = posterior.reshape((len(np.unique(T_list)),len(np.unique(N_list))))
    grid_indices = np.arange(posterior.size)
    N_samples = 100000
    sampled_indices = np.random.choice(grid_indices, size=N_samples, p=posterior)
    i_sampled, j_sampled = np.unravel_index(sampled_indices, posterior_grid.shape)
    # Convert to physical N and T values
    T_sampled_values = np.unique(T_list)[i_sampled]
    N_sampled_values = np.unique(N_list)[j_sampled]

    # Compute upper limits (e.g. 95% for N)
    N_95_upper = np.percentile(N_sampled_values, 95)
    print(f"95% upper limit on N: {N_95_upper:.1f}")
    nx = len(np.unique(T_list))
    ny = len(np.unique(N_list))

    Rdisk_grid = Rdisks.reshape(ny, nx)
    Z = chis.reshape(ny, nx)  # NOTE: imshow expects shape (rows=ny, cols=nx)
    Z = np.min(Z)/Z

    map_extent = [T_list[0], T_list[-1],
                  N_list[0], N_list[-1]]
    # Now plot with imshow
    m = plt.imshow(np.log(Z), extent=map_extent, origin='lower', aspect='auto',
               cmap='magma')
    plt.colorbar(m, label=r'$\chi^2_{min}/\chi^2$')

    Rdisk_levels = np.linspace(0, 2, 5)
    Rdisk_contours = plt.contour(Rdisk_grid, levels=Rdisk_levels, origin='lower',
                                 extent=map_extent, colors='white', zorder=2)
    fmt = ticker.LogFormatterMathtext()
    fmt.create_dummy_axis()
    plt.clabel(Rdisk_contours, Rdisk_contours.levels, fmt=fmt)



    plt.xlabel('T [K]')
    plt.ylabel('N [log(cm-2)]')
    plt.savefig('Figures/CHI2MAP.pdf')
    plt.show()
    m = plt.imshow(posterior_grid.T, extent=map_extent, origin='lower', aspect='auto')
    plt.colorbar(m, label=r'$\chi^2_{min}/\chi^2$')
    plt.xlabel('T [K]')
    plt.ylabel('N [log(cm-2)]')
    plt.show()

    T_bins = np.linspace(min(T_sampled_values), max(T_sampled_values), len(np.unique(T_sampled_values))+1)
    N_bins = np.linspace(min(N_sampled_values), max(N_sampled_values), len(np.unique(N_sampled_values))+1)

    # Compute 2D histogram
    H, T_edges, N_edges = np.histogram2d(T_sampled_values, N_sampled_values, bins=[T_bins, N_bins], density=True)

    # Plot with imshow
    plt.pcolormesh(T_edges, N_edges, H.T, cmap='inferno', shading='auto')
    plt.axhline(N_95_upper, c='white')
    plt.text(1200, N_95_upper + 0.3, f'Upper Limit = {N_95_upper:.1f}', fontsize=14, color='white', ha='center', va='bottom')
    plt.xlabel(r'T$_K$ in K')
    plt.ylabel(r'log10(N({})) in cm$^{{-2}}$'.format(mol))
    plt.colorbar(label="Sample Density")

    plt.tight_layout()
    plt.savefig('upper_{}_{}.pdf'.format(mol, Source), bbox_inches='tight')
    plt.show()