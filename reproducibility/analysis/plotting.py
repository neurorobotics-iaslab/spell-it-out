import matplotlib.pyplot as plt
import mne
import numpy as np


def plot_erd_grid(tfr, bands, time_windows, vlim=(-0.5, 0.5), cmap='RdBu_r', title='', cbar_label='ERD / ERS'):
    """
    Genera una griglia di topoplot con una SINGOLA colorbar orizzontale in basso.
    """
    n_bands = len(bands)
    n_times = len(time_windows)
    
    if not isinstance(time_windows, dict):
        time_windows = {f"{tmin}s - {tmax}s": (tmin, tmax) for tmin, tmax in time_windows}
    # Prepariamo la figura
    fig, axes = plt.subplots(n_bands, n_times, figsize=(n_times * 3, n_bands * 2.5 + 0))
    
    if n_bands == 1 and n_times == 1: axes = np.array([[axes]])
    elif n_bands == 1: axes = axes[np.newaxis, :]
    elif n_times == 1: axes = axes[:, np.newaxis]

    ims = []
    plotted_values = []
    if vlim is None:
        vlim_auto = True
        vlim = (None, None)
    else:
        vlim_auto = False

    for i, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        for j, (time_name, (tmin, tmax)) in enumerate(time_windows.items()):
            ax = axes[i, j]
            
            freq_mask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
            time_mask = (tfr.times >= tmin) & (tfr.times <= tmax)
            
            data_subset = tfr.data[:, freq_mask, :][:, :, time_mask]
            mean_data_per_channel = np.nanmean(data_subset, axis=(1, 2))
            plotted_values.append(mean_data_per_channel)
            
            im, _ = mne.viz.plot_topomap(
                mean_data_per_channel, 
                tfr.info, 
                axes=ax, 
                cmap=cmap, 
                vlim=vlim,
                contours=4, 
                extrapolate='head', 
                show=False
            )
            ims.append(im)
            if j == 0:
                ax.text(-0.25, 0.5, f"{band_name}\n{fmin}-{fmax} Hz", 
                        transform=ax.transAxes, va='center', ha='right', 
                        fontsize=11, fontweight='bold')
                
            if i == 0:
                ax.set_title(f"{time_name}", fontsize=11, pad=10,fontweight='bold')

    if vlim_auto:
        vlim = max([im.get_clim()[1] for im in ims])
        for im in ims: im.set_clim((-vlim, vlim))
    # --- COLORBAR ORIZZONTALE  IN BASSO ---
    # orientation='horizontal' e location='bottom'
    # fraction e pad controllano quanto spazio occupa rispetto ai plot
    # Explicit limits clip out-of-range values. Tell the reader about that
    # clipping with the standard triangular colorbar extensions.
    colorbar_extend = 'neither'
    if not vlim_auto:
        values = np.concatenate(plotted_values)
        lower_clipped = vlim[0] is not None and np.nanmin(values) < vlim[0]
        upper_clipped = vlim[1] is not None and np.nanmax(values) > vlim[1]
        if lower_clipped and upper_clipped:
            colorbar_extend = 'both'
        elif lower_clipped:
            colorbar_extend = 'min'
        elif upper_clipped:
            colorbar_extend = 'max'

    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.96)
    fig.subplots_adjust(top=0.90, bottom=0.16, left=0.12, right=0.95, hspace=0.1, wspace=0.1)

    # Use a dedicated axis rather than letting ``fig.colorbar(..., ax=...)``
    # resize/overlap the last row when the bottom margin is compact.
    cbar_ax = fig.add_axes([0.27, 0.065, 0.50, 0.025])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                        extend=colorbar_extend)
    cbar.set_label(cbar_label, fontsize=11, labelpad=10)

    return fig

def topoplot(info, data, title="",ax = None,vlim=None):
    if ax is None: 
        fig, ax = plt.subplots(figsize=(6, 6))
    mne.viz.plot_topomap(data, info, axes=ax, show=False,cmap='viridis',vlim=vlim)
    ax.set_title(title)
