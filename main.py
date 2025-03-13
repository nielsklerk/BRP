import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


with open('ModelData.pkl', 'rb') as f:
    df = pickle.load(f)

plt.close('all')
fig = plt.figure(figsize=(14, 14))
gs = fig.add_gridspec(5, 5, hspace=0, wspace=0)
axes = gs.subplots(sharex='col', sharey='row').flatten()
species = ['C2H2_H', 'CH4_H', 'CO', 'CO2_H', 'HCN_H', 'NH3_H', 'NO_H', 'OH', 'S', 'o-H2', 'o-H2O', 'p-H2', 'p-H2O',
           'Ion']
for specie in species:
    for i, row in df.iterrows():
        n_row, n_col = divmod(i, 5)
        axes[i].plot(row['Prodimo Wavelength'], row[specie], label=row['Model Name'])
        axes[i].set_xticks([10, 20])
        axes[i].set_yticks(np.linspace(0, 0.008, 5))
        axes[i].text(6, 0.008, f"C={row['C Value']} O={row['O Value']} CO={row['CO Value']}", fontsize=9, color='red',
                     weight='bold')
        axes[i].set_ylim([0, 0.01])
        # Remove inner tick labels
        if n_row < 4:
            axes[i].set_xticklabels([])  # Hide x-tick labels except last row
        if n_col > 0:
            axes[i].set_yticklabels([])  # Hide y-tick labels except first column

        # Set labels only on the outside
        if n_row == 4:
            axes[i].set_xlabel(r"Wavelength ($\mu$m)")  # Bottom row
        if n_col == 0:
            axes[i].set_ylabel("Flux (Jy)")  # Leftmost column
    fig.suptitle(specie, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()