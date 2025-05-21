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


def check_chi(args):
    i, index = args
    slab_model = rs.read_slab(f'NO/NO_{index}.fits.gz', verbose=True)
    min_chi = min(rs.red_chi2_slab(slab_model, spectra, mask=mask, overlap=True))
    return i, min_chi


def mask_regions(wavelength, regions):
    mask = np.ones_like(wavelength, dtype=bool)
    for region in regions:
        mask[np.logical_and(wavelength > region[0], wavelength < region[1])] = False
    return mask


for Source in ['V1094Sco']:

    file = f'FullSpectrum_CS_{Source}.p'
    data = pickle.load(open(file, 'rb'))
    wavelength = data['Wavelength']
    flux_cont_sub = data['CSFlux']
    clip_min, clip_max = 4.9, 6.5  # micron

    clip_cnd = ((wavelength >= clip_min) & (wavelength <= clip_max))
    total = flux_cont_sub[clip_cnd].copy()
    # for mol in ['H2O', 'CO']:
    #     file = f'{Source}4_9_6_3/{mol}_best_fit.p'
    #     data = pickle.load(open(file, 'rb'))
    #     o_w = data['o_w']
    #     # o_w = np.nan_to_num(o_w, nan=0)
    #     m_f = data['m_f']
    #     m_f = np.nan_to_num(m_f, nan=0)
    #     # plt.plot(o_w, mask_regions(o_w, 1000*m_f*data['Rdisk_best']**2, artefact_regions), label=mol)
    #     total -= m_f * data['Rdisk_best'] ** 2
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



if __name__ == '__main__':
    chis = np.zeros_like(index_list)
    tasks = [(int(k - 1), f'{int(k):04d}') for k in index_list]
    with Pool(11) as pool:
        results = pool.map(check_chi, tasks)
    for i, chi in results:
        chis[i] = chi

    nx = len(np.unique(T_list))
    ny = len(np.unique(N_list))

    Z = chis.reshape(ny, nx)  # NOTE: imshow expects shape (rows=ny, cols=nx)

    # Now plot with imshow
    plt.imshow(Z, extent=(T_list.min(), T_list.max(), N_list.min(), N_list.max()), origin='lower', aspect='auto',
               cmap='magma')
    plt.colorbar(label='Reduced Chi-squared')
    plt.xlabel('T [K]')
    plt.ylabel('N [log(cm-2)]')
    plt.savefig('Figures/CHI2MAP.pdf')
    plt.show()