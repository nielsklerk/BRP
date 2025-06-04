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


def check_chi(args):
    i, index = args
    slab_model = rs.read_slab(f'NO/NO_{index}.fits.gz', verbose=True)
    min_chi = min(rs.red_chi2_slab(slab_model, spectra, mask=mask, overlap=True))
    return i, min_chi

def corr(args):
    i, index = args
    slab_model = rs.read_slab(f'NO/NO_{index}.fits.gz', verbose=True)
    min_chi = min(rs.red_chi2_slab(slab_model, spectra, mask=mask, overlap=True))
    return i, min_chi


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
clip_min, clip_max = 4.9, 6.5  # micron

clip_cnd = ((wavelength >= clip_min) & (wavelength <= clip_max))
total = flux_cont_sub[clip_cnd].copy()
for mol in ['H2O', 'CO']:
    file = f'{Source}4_9_6_3/{mol}_best_fit.p'
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
# for index in index_list:
#     print(f'Index: {index}, T_list: {T_list[int(index)]}, N_list: {N_list[int(index)]}')
test_slab = rs.read_slab(f'NO/NO_{int(index_list[(T_list==800)&(N_list==14)]):04d}.fits.gz', verbose=True)


Rdisk = 3
distance = [155.20, 156.27, 152.44][model_index]
vr = [-3300, -1400, 2200][model_index]

test_slab.convolve(R=3000, overlap=True, NLTE=False, vr=vr)

# area=np.pi*(Rdisk*au/distance/pc)**2
# modelSpec=spectres(spectra[:,0],c/test_slab.convOverlapFreq[::-1]*1e-3,test_slab.convOverlapLTE[::-1]*1e23,verbose=False,fill=0.0)
# norm_modelSpec = np.trapezoid(modelSpec, wavelength[clip_cnd])
# fig,ax=plt.subplots(figsize=(12,4))
# ax.step(wavelength[clip_cnd], modelSpec*area*1000)
# ax.step(wavelength[clip_cnd], total*1000)
# ax.set_xlim([5.1, 5.6])
# ax.set_ylim([-5, 15])
# plt.show()



norm_total = total / np.trapezoid(total, wavelength[clip_cnd])
def crosscorr(index):
    test_slab = rs.read_slab(f'NO/NO_{int(index):04d}.fits.gz', verbose=True)
    test_slab.convolve(R=3000, overlap=True, NLTE=False, vr=vr)
    modelSpec = spectres(spectra[:, 0], c / test_slab.convOverlapFreq[::-1] * 1e-3,
                         test_slab.convOverlapLTE[::-1] * 1e23, verbose=False, fill=0.0)
    norm_modelSpec = modelSpec / np.trapezoid(modelSpec, wavelength[clip_cnd])
    # norm_modelSpec = modelSpec * area
    cc_mid = correlate(norm_modelSpec, norm_total, mode='full')[len(norm_modelSpec) - 1]
    return cc_mid

if __name__ == '__main__':
    cc = np.zeros_like(index_list)
    tasks = [f'{int(k):04d}' for k in index_list]
    with Pool(11) as pool:
        results = pool.map(crosscorr, tasks)
    for i, cc_mid in enumerate(results):
        cc[i] = cc_mid

    nx = len(np.unique(T_list))
    ny = len(np.unique(N_list))

    Z = cc.reshape(ny, nx)  # NOTE: imshow expects shape (rows=ny, cols=nx)

    # Now plot with imshow
    plt.imshow(Z, extent=(T_list.min(), T_list.max(), N_list.min(), N_list.max()), origin='lower', aspect='auto',
               cmap='magma')
    plt.colorbar(label='CrossCorrelation')
    plt.xlabel('T [K]')
    plt.ylabel('N [log(cm-2)]')
    # plt.savefig('Figures/CHI2MAP.pdf')
    plt.show()


# if __name__ == '__main__':
#     chis = np.zeros_like(index_list)
#     tasks = [(int(k - 1), f'{int(k):04d}') for k in index_list]
#     with Pool(11) as pool:
#         results = pool.map(check_chi, tasks)
#     for i, chi in results:
#         chis[i] = chi
#
#     log_likelihood = -0.5 * chis
#     log_prior = np.zeros_like(log_likelihood)
#
#     # Compute unnormalized log-posterior
#     log_posterior = log_likelihood + log_prior
#     posterior = np.exp(log_posterior - np.max(log_posterior))  # prevent underflow
#     posterior /= np.sum(posterior)
#     posterior_grid = posterior.reshape((len(np.unique(T_list)),len(np.unique(N_list))))
#     grid_indices = np.arange(posterior.size)
#     N_samples = 100000
#     sampled_indices = np.random.choice(grid_indices, size=N_samples, p=posterior)
#     i_sampled, j_sampled = np.unravel_index(sampled_indices, posterior_grid.shape)
#     # Convert to physical N and T values
#     T_sampled_values = np.unique(T_list)[i_sampled]
#     N_sampled_values = np.unique(N_list)[j_sampled]
#
#     # Compute upper limits (e.g. 95% for N)
#     N_95_upper = np.percentile(N_sampled_values, 95)
#     print(f"95% upper limit on N: {N_95_upper:.2e}")
#     nx = len(np.unique(T_list))
#     ny = len(np.unique(N_list))
#
#     Z = chis.reshape(ny, nx)  # NOTE: imshow expects shape (rows=ny, cols=nx)
#
#     # Now plot with imshow
#     plt.imshow(Z, extent=(T_list.min(), T_list.max(), N_list.min(), N_list.max()), origin='lower', aspect='auto',
#                cmap='magma')
#     plt.colorbar(label='Reduced Chi-squared')
#     plt.xlabel('T [K]')
#     plt.ylabel('N [log(cm-2)]')
#     plt.savefig('Figures/CHI2MAP.pdf')
#     plt.show()
#     plt.subplot(1, 2, 2)
#     plt.hexbin(T_sampled_values, N_sampled_values,  gridsize=20, cmap='inferno')
#     plt.xlabel("Column Density N")
#     plt.ylabel("Temperature T")
#     plt.colorbar(label="Sample Density")
#
#     plt.tight_layout()
#     plt.show()