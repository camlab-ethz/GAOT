"""
Plotting utilities for GAOT results visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Tuple
import matplotlib.colors as mcolors

import matplotlib

########################################################
# Plotting settings
########################################################
# Dark colors
C_BLACK = '#000000'
C_WHITE = '#ffffff'
C_BLUE = '#093691'
C_RED = '#911b09'
C_BLACK_BLUEISH = '#011745'
C_BLACK_REDDISH = '#380801'
C_WHITE_BLUEISH = '#dce5f5'
C_WHITE_REDDISH = '#f5dcdc'
# Bright colors
C_BRIGHT_PURPLE = '#7f00ff'   
C_BRIGHT_PINK   = '#ff00ff'   
C_BRIGHT_ORANGE = '#ff7700'   
C_BRIGHT_YELLOW = '#ffdd00'   
C_BRIGHT_GREEN  = '#00ee00'   
C_BRIGHT_CYAN   = '#00ffff'   
C_BRIGHT_BLUE   = '#0f00ff'   
CMAP_BWR = matplotlib.colors.LinearSegmentedColormap.from_list(
  'blue_white_red',
  [C_BLACK_BLUEISH, C_BLUE, C_WHITE, C_RED, C_BLACK_REDDISH],
  N=200,
)
CMAP_WRB = matplotlib.colors.LinearSegmentedColormap.from_list(
  'white_red_black',
  [C_WHITE, C_RED, C_BLACK],
  N=200,
)
# Scatter settings
SCATTER_SETTINGS = dict(marker='s', s=1, alpha=1, linewidth=0)
HATCH_SETTINGS = dict(facecolor='#b8b8b8', edgecolor='#4f4f4f', linewidth=.0)

########################################################
# Plotting functions
########################################################
def plot_estimates(
    u_inp: np.ndarray,
    u_gtr: np.ndarray,
    u_prd: np.ndarray,
    x_inp: np.ndarray,
    x_out: np.ndarray,
    symmetric: Union[bool, List[bool]] = True,
    names: Optional[List[str]] = None,
    domain: Tuple[List[float], List[float]] = ([-1, -1], [1, 1]),
    colorbar_type: str = "light",
    show_error: bool = True
) -> plt.Figure:
    """
    Plots input data, ground-truth, model predictions, and optionally absolute errors over a 2D domain.

    This function creates a figure with three or four panels (columns) for each variable:
    1) Input data,
    2) Ground-truth values,
    3) Model predictions,
    4) Absolute error (|ground-truth - prediction|) - optional based on show_error parameter.

    A horizontal colorbar is provided for each column, showing the data range used for coloring.
    
    Parameters
    ----------
    u_inp : np.ndarray
        The input data array of shape (N_inp, n_input_vars), where:
          - N_inp is the number of input points.
          - n_input_vars is the number of input variables (e.g., different physical quantities).
    u_gtr : np.ndarray
        The ground-truth data array of shape (N_out, n_output_vars). N_out can differ from N_inp
        if the input and output grids do not match.
    u_prd : np.ndarray
        The model-predicted data array, same shape as `u_gtr` (i.e., (N_out, n_output_vars)).
        This is compared against `u_gtr` to compute the absolute error.
    x_inp : np.ndarray
        The (x, y) coordinates of each input point, shape (N_inp, 2).
        Used for the scatter plot of `u_inp`.
    x_out : np.ndarray
        The (x, y) coordinates for the output/ground-truth grid, shape (N_out, 2).
        Used for the scatter plots of `u_gtr`, `u_prd`, and their absolute error.
    symmetric : bool or list of bool, optional
        Whether to use a symmetric color scale (colormap) for each variable. 
        If True, the color limits are set to [-vmax, +vmax], where vmax is 
        the maximum absolute value across data samples for that variable. 
        If a list of booleans is provided, each element corresponds to one variable.
    names : list of str, optional
        A list of variable names (of length n_vars) used as labels on the vertical axis.
        If None, default labels such as "Variable 00", "Variable 01", etc., are used.
    domain : tuple of list, optional
        Defines the displayed plotting region as ([x_min, y_min], [x_max, y_max]).
        Defaults to ([0, 0], [1, 1]). The background hatch pattern will fill this region.
    colorbar_type: str, optional
        The type of colorbar to use. Can be "light" or "dark". Defaults to "light".
    show_error: bool, optional
        Whether to show the absolute error column. Defaults to True.
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the subplots. Each variable has one row in the figure,
        and there are three or four columns of scatter plots: input, ground-truth, prediction,
        and optionally absolute error. Each column is accompanied by a horizontal colorbar.

    Notes
    -----
    - Internally, the function arranges subplots for each variable in a two-row layout:
      the top row is for the actual scatter plots, and the bottom row hosts the colorbars.
    - The absolute error is plotted as |u_gtr - u_prd|.
    - Use the returned figure object to further customize, save, or display the figure.

    Examples
    --------
    >>> import numpy as np
    >>> # Assume we have two variables (n_vars = 2)
    >>> # Input grid has 50 points, output grid has 100 points
    >>> x_inp = np.random.rand(50, 2)
    >>> x_out = np.random.rand(100, 2)
    >>> u_inp = np.random.randn(50, 2)
    >>> u_gtr = np.random.randn(100, 2)
    >>> u_prd = u_gtr + 0.1 * np.random.randn(100, 2)
    >>> fig = plot_estimates(
    ...     u_inp=u_inp,
    ...     u_gtr=u_gtr,
    ...     u_prd=u_prd,
    ...     x_inp=x_inp,
    ...     x_out=x_out,
    ...     symmetric=True,
    ...     names=["Temperature", "Concentration"],
    ...     domain=([0, 0], [1, 1])
    ... )
    >>> fig.tight_layout()
    >>> fig.show()
    """
    _HEIGHT_PER_ROW = 1.9
    _HEIGHT_MARGIN = .2
    _SCATTER_SETTINGS = SCATTER_SETTINGS.copy()
    _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * .4 * _HEIGHT_PER_ROW
    _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * 128 / (x_inp.shape[0] ** .5)

    n_vars = u_gtr.shape[-1]
    if isinstance(symmetric, bool):
        symmetric = [symmetric] * n_vars

    # Calculate number of columns and adjust figsize accordingly
    n_cols = 4 if show_error else 3
    base_width = 8.6  # Original width for 4 columns
    figsize = (base_width * n_cols / 4.0, _HEIGHT_PER_ROW*n_vars+_HEIGHT_MARGIN)
    fig = plt.figure(figsize=figsize)
    g_fig = fig.add_gridspec(
        nrows=n_vars,
        ncols=1,
        wspace=0,
        hspace=0,
    )

    figs = []
    for ivar in range(n_vars):
        figs.append(fig.add_subfigure(g_fig[ivar], frameon=False))
    # Add axes
    axs_inp = []
    axs_gtr = []
    axs_prd = []
    axs_err = []
    axs_cb_inp = []
    axs_cb_out = []
    axs_cb_err = []
    for ivar in range(n_vars):
        g = figs[ivar].add_gridspec(
        nrows=2,
        ncols=n_cols,
        height_ratios=[1, .05],
        wspace=0.20,
        hspace=0.05,
        )
        axs_inp.append(figs[ivar].add_subplot(g[0, 0]))
        axs_gtr.append(figs[ivar].add_subplot(g[0, 1]))
        axs_prd.append(figs[ivar].add_subplot(g[0, 2]))
        if show_error:
            axs_err.append(figs[ivar].add_subplot(g[0, 3]))
        else:
            axs_err.append(None)  # Placeholder to maintain indexing
        
        axs_cb_inp.append(figs[ivar].add_subplot(g[1, 0]))
        if show_error:
            axs_cb_out.append(figs[ivar].add_subplot(g[1, 1:3]))
            axs_cb_err.append(figs[ivar].add_subplot(g[1, 3]))
        else:
            axs_cb_out.append(figs[ivar].add_subplot(g[1, 1:3]))  # Spans to the end
            axs_cb_err.append(None)  # Placeholder
    # Settings
    all_axes = [axs_inp, axs_gtr, axs_prd]
    if show_error:
        all_axes.append(axs_err)
    for ax in [ax for axs in all_axes for ax in axs if ax is not None]:
        ax: plt.Axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([domain[0][0], domain[1][0]])
        ax.set_ylim([domain[0][1], domain[1][1]])
        ax.fill_between(
        x=[domain[0][0], domain[1][0]], y1=domain[0][1], y2=domain[1][1],
        **HATCH_SETTINGS,
        )

    # Get prediction error
    u_err = (u_gtr - u_prd)

    # Choose colormap based on colorbar_type
    if colorbar_type == "light":
        cmap_symmetric = plt.cm.jet
        cmap_asymmetric = plt.cm.jet
    else:
        cmap_symmetric = CMAP_BWR
        cmap_asymmetric = CMAP_WRB

    # Loop over variables
    for ivar in range(n_vars):
        # Get ranges
        vmax_inp = np.max(u_inp[:, ivar])
        vmax_gtr = np.max(u_gtr[:, ivar])
        vmax_prd = np.max(u_prd[:, ivar])
        vmax_out = max(vmax_gtr, vmax_prd)
        vmin_inp = np.min(u_inp[:, ivar])
        vmin_gtr = np.min(u_gtr[:, ivar])
        vmin_prd = np.min(u_prd[:, ivar])
        vmin_out = min(vmin_gtr, vmin_prd)
        abs_vmax_inp = max(np.abs(vmax_inp), np.abs(vmin_inp))
        abs_vmax_out = max(np.abs(vmax_out), np.abs(vmin_out))

        # Plot input
        h = axs_inp[ivar].scatter(
        x=x_inp[:, 0],
        y=x_inp[:, 1],
        c=u_inp[:, ivar],
        cmap=(cmap_symmetric if symmetric[ivar] else cmap_asymmetric),
        vmax=(abs_vmax_inp if symmetric[ivar] else vmax_inp),
        vmin=(-abs_vmax_inp if symmetric[ivar] else vmin_inp),
        **_SCATTER_SETTINGS,
        )
        cb = plt.colorbar(h, cax=axs_cb_inp[ivar], orientation='horizontal')
        cb.formatter.set_powerlimits((-0, 0))
        # Plot ground truth
        h = axs_gtr[ivar].scatter(
        x=x_out[:, 0],
        y=x_out[:, 1],
        c=u_gtr[:, ivar],
        cmap=(cmap_symmetric if symmetric[ivar] else cmap_asymmetric),
        vmax=(abs_vmax_out if symmetric[ivar] else vmax_out),
        vmin=(-abs_vmax_out if symmetric[ivar] else vmin_out),
        **_SCATTER_SETTINGS,
        )
        # Plot estimate
        h = axs_prd[ivar].scatter(
        x=x_out[:, 0],
        y=x_out[:, 1],
        c=u_prd[:, ivar],
        cmap=(cmap_symmetric if symmetric[ivar] else cmap_asymmetric),
        vmax=(abs_vmax_out if symmetric[ivar] else vmax_out),
        vmin=(-abs_vmax_out if symmetric[ivar] else vmin_out),
        **_SCATTER_SETTINGS,
        )
        cb = plt.colorbar(h, cax=axs_cb_out[ivar], orientation='horizontal')
        cb.formatter.set_powerlimits((-0, 0))

        # Plot error (only if show_error is True)
        if show_error:
            h = axs_err[ivar].scatter(
            x=x_out[:, 0],
            y=x_out[:, 1],
            c=np.abs(u_err[:, ivar]),
            cmap=cmap_asymmetric,
            vmin=0,
            vmax=np.max(np.abs(u_err[:, ivar])),
            **_SCATTER_SETTINGS,
            )
            cb = plt.colorbar(h, cax=axs_cb_err[ivar], orientation='horizontal')
            cb.formatter.set_powerlimits((-0, 0))

    # Set titles
    axs_inp[0].set(title='Input')
    axs_gtr[0].set(title='Ground-truth')
    axs_prd[0].set(title='Model estimate')
    if show_error:
        axs_err[0].set(title='Absolute error')

    # Set variable names
    for ivar in range(n_vars):
        label = names[ivar] if names else f'Variable {ivar:02d}'
        axs_inp[ivar].set(ylabel=label);

    # Rotate colorbar tick labels
    cb_axes = [axs_cb_inp, axs_cb_out]
    if show_error:
        cb_axes.append(axs_cb_err)
    for ax in [ax for axs in cb_axes for ax in axs if ax is not None]:
        ax: plt.Axes
        ax.xaxis.get_offset_text().set(size=8)
        ax.xaxis.set_tick_params(labelsize=8)

    return fig


def create_sequential_animation(gt_sequence: np.ndarray, pred_sequence: np.ndarray,
                               coords: np.ndarray, save_path: str,
                               input_data: np.ndarray = None,
                               time_values: List[float] = None,
                               interval: int = 500, symmetric: Union[bool, List[bool]] = True,
                               domain: Tuple[List[float], List[float]] = None,
                               names: List[str] = None,
                               colorbar_type: str = "light",
                               show_error: bool = True) -> None:
    """
    Create animation comparing input, ground truth and prediction sequences.
    Uses 3 or 4-column layout identical to plot_estimates: Input | Ground Truth | Prediction | [Error]
    
    Args:
        gt_sequence: Ground truth sequence [n_timesteps, n_points, n_channels]
        pred_sequence: Prediction sequence [n_timesteps, n_points, n_channels]
        coords: Coordinates [n_points, coord_dim]
        save_path: Path to save animation
        input_data: Input data [n_points, n_channels] (static, shown in first column)
        time_values: List of time values for each frame
        interval: Interval between frames in milliseconds
        symmetric: Whether to use symmetric colorscale (bool or list of bool)
        domain: Plotting domain ([x_min, y_min], [x_max, y_max])
        names: Variable names for each channel
        colorbar_type: The type of colorbar to use ("light" or "dark")
        show_error: Whether to show the absolute error column (defaults to True)
    """
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("Matplotlib animation not available")
        return
    
    if coords.shape[1] != 2:
        print("Animation currently only supports 2D coordinates")
        return
    
    n_timesteps, n_points, n_channels = gt_sequence.shape
    
    _HEIGHT_PER_ROW = 1.9
    _HEIGHT_MARGIN = .2
    _SCATTER_SETTINGS = SCATTER_SETTINGS.copy()
    _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * .4 * _HEIGHT_PER_ROW
    _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * 128 / (coords.shape[0] ** .5)
    
    if isinstance(symmetric, bool):
        symmetric = [symmetric] * n_channels
    
    if colorbar_type == "light":
        cmap_symmetric = plt.cm.jet
        cmap_asymmetric = plt.cm.jet
    else:
        cmap_symmetric = CMAP_BWR
        cmap_asymmetric = CMAP_WRB
    
    # Calculate number of columns and adjust figsize accordingly
    n_cols = 4 if show_error else 3
    base_width = 8.6  # Original width for 4 columns
    figsize = (base_width * n_cols / 4.0, _HEIGHT_PER_ROW*n_channels+_HEIGHT_MARGIN)
    fig = plt.figure(figsize=figsize)
    g_fig = fig.add_gridspec(
        nrows=n_channels,
        ncols=1,
        wspace=0,
        hspace=0,
    )

    figs = []
    for ivar in range(n_channels):
        figs.append(fig.add_subfigure(g_fig[ivar], frameon=False))
    
    if domain is not None:
        plot_domain = domain
    else:
        plot_domain = ([coords[:, 0].min(), coords[:, 1].min()], 
                      [coords[:, 0].max(), coords[:, 1].max()])
    
    scatter_objects = {'inp': [], 'gt': [], 'pred': [], 'error': []}
    axes_inp = []
    axes_gt = []
    axes_pred = []
    axes_err = []
    axes_cb_inp = []
    axes_cb_gt = []
    axes_cb_err = []
    
    # Add axes for each channel following plot_estimates pattern exactly
    for ivar in range(n_channels):
        g = figs[ivar].add_gridspec(
            nrows=2,
            ncols=n_cols,
            height_ratios=[1, .05],
            wspace=0.20,
            hspace=0.05,
        )
        axes_inp.append(figs[ivar].add_subplot(g[0, 0]))
        axes_gt.append(figs[ivar].add_subplot(g[0, 1]))
        axes_pred.append(figs[ivar].add_subplot(g[0, 2]))
        if show_error:
            axes_err.append(figs[ivar].add_subplot(g[0, 3]))
        else:
            axes_err.append(None)  # Placeholder
            
        axes_cb_inp.append(figs[ivar].add_subplot(g[1, 0]))
        if show_error:
            axes_cb_gt.append(figs[ivar].add_subplot(g[1, 1:3]))  # Spans 2 columns like plot_estimates
            axes_cb_err.append(figs[ivar].add_subplot(g[1, 3]))
        else:
            axes_cb_gt.append(figs[ivar].add_subplot(g[1, 1:3]))  # Spans to the end
            axes_cb_err.append(None)  # Placeholder
    
    # Settings for all axes (same as plot_estimates)
    all_axes = [axes_inp, axes_gt, axes_pred]
    if show_error:
        all_axes.append(axes_err)
    for ax in [ax for axs in all_axes for ax in axs if ax is not None]:
        ax: plt.Axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([plot_domain[0][0], plot_domain[1][0]])
        ax.set_ylim([plot_domain[0][1], plot_domain[1][1]])
        ax.fill_between(
            x=[plot_domain[0][0], plot_domain[1][0]], 
            y1=plot_domain[0][1], y2=plot_domain[1][1],
            **HATCH_SETTINGS,
        )
    
    u_err_0 = (gt_sequence[0] - pred_sequence[0])
    
    for ivar in range(n_channels):
        gt_all = gt_sequence[:, :, ivar]
        pred_all = pred_sequence[:, :, ivar]
        
        vmax_gtr = np.max(gt_all)
        vmax_prd = np.max(pred_all)
        vmax_out = max(vmax_gtr, vmax_prd)
        vmin_gtr = np.min(gt_all)
        vmin_prd = np.min(pred_all)
        vmin_out = min(vmin_gtr, vmin_prd)
        abs_vmax_out = max(np.abs(vmax_out), np.abs(vmin_out))
        
        if input_data is not None:
            vmax_inp = np.max(input_data[:, ivar])
            vmin_inp = np.min(input_data[:, ivar])
            abs_vmax_inp = max(np.abs(vmax_inp), np.abs(vmin_inp))
            
            h_inp = axes_inp[ivar].scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                c=input_data[:, ivar],
                cmap=(cmap_symmetric if symmetric[ivar] else cmap_asymmetric),
                vmax=(abs_vmax_inp if symmetric[ivar] else vmax_inp),
                vmin=(-abs_vmax_inp if symmetric[ivar] else vmin_inp),
                **_SCATTER_SETTINGS,
            )
            scatter_objects['inp'].append(h_inp)
            cb_inp = plt.colorbar(h_inp, cax=axes_cb_inp[ivar], orientation='horizontal')
            cb_inp.formatter.set_powerlimits((-0, 0))
        else:
            h_inp = axes_inp[ivar].scatter([], [], **_SCATTER_SETTINGS)
            scatter_objects['inp'].append(h_inp)
        
        # Plot ground truth
        h_gt = axes_gt[ivar].scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            c=gt_sequence[0, :, ivar],
            cmap=(cmap_symmetric if symmetric[ivar] else cmap_asymmetric),
            vmax=(abs_vmax_out if symmetric[ivar] else vmax_out),
            vmin=(-abs_vmax_out if symmetric[ivar] else vmin_out),
            **_SCATTER_SETTINGS,
        )
        scatter_objects['gt'].append(h_gt)
        cb_gt = plt.colorbar(h_gt, cax=axes_cb_gt[ivar], orientation='horizontal')
        cb_gt.formatter.set_powerlimits((-0, 0))
        
        # Plot prediction
        h_pred = axes_pred[ivar].scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            c=pred_sequence[0, :, ivar],
            cmap=(cmap_symmetric if symmetric[ivar] else cmap_asymmetric),
            vmax=(abs_vmax_out if symmetric[ivar] else vmax_out),
            vmin=(-abs_vmax_out if symmetric[ivar] else vmin_out),
            **_SCATTER_SETTINGS,
        )
        scatter_objects['pred'].append(h_pred)
        cb_pred = plt.colorbar(h_pred, cax=axes_cb_gt[ivar], orientation='horizontal')
        cb_pred.formatter.set_powerlimits((-0, 0))
        
        if show_error:
            h_err = axes_err[ivar].scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                c=np.abs(u_err_0[:, ivar]),
                cmap=cmap_asymmetric,
                vmin=0,
                vmax=np.max(np.abs(gt_sequence[:, :, ivar] - pred_sequence[:, :, ivar])),
                **_SCATTER_SETTINGS,
            )
            scatter_objects['error'].append(h_err)
            cb_err = plt.colorbar(h_err, cax=axes_cb_err[ivar], orientation='horizontal')
            cb_err.formatter.set_powerlimits((-0, 0))
        else:
            scatter_objects['error'].append(None)  # Placeholder
    
    axes_inp[0].set(title='Input')
    axes_gt[0].set(title='Ground truth')
    axes_pred[0].set(title='Prediction')
    if show_error:
        axes_err[0].set(title='Absolute error')
    
    for ivar in range(n_channels):
        label = names[ivar] if names and ivar < len(names) else f'Variable {ivar:02d}'
        axes_inp[ivar].set(ylabel=label)
    
    cb_axes = [axes_cb_inp, axes_cb_gt]
    if show_error:
        cb_axes.append(axes_cb_err)
    for ax in [ax for axs in cb_axes for ax in axs if ax is not None]:
        ax: plt.Axes
        ax.xaxis.get_offset_text().set(size=8)
        ax.xaxis.set_tick_params(labelsize=8)
    
    def animate(frame):
        """Update function for animation."""
        for ivar in range(n_channels):
            # Update ground truth
            scatter_objects['gt'][ivar].set_array(gt_sequence[frame, :, ivar])
            
            # Update prediction
            scatter_objects['pred'][ivar].set_array(pred_sequence[frame, :, ivar])
            
            # Update error (only if show_error is True)
            if show_error and scatter_objects['error'][ivar] is not None:
                error = np.abs(gt_sequence[frame, :, ivar] - pred_sequence[frame, :, ivar])
                scatter_objects['error'][ivar].set_array(error)
        
        # if time_values and frame < len(time_values):
        #     fig.suptitle(f'Time: {time_values[frame]:.3f}')
        # else:
        #     fig.suptitle(f'Time step: {frame}')
        
        all_scatters = []
        for key in scatter_objects:
            all_scatters.extend([obj for obj in scatter_objects[key] if obj is not None])
        return all_scatters
    
    anim = FuncAnimation(fig, animate, frames=n_timesteps, 
                        interval=interval, blit=False, repeat=True)
    
    print(f"Saving sequential animation to {save_path}...")
    try:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000//interval, dpi=150)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=1000//interval, dpi=150)
        else:
            # Default to gif
            save_path_gif = save_path + '.gif'
            anim.save(save_path_gif, writer='pillow', fps=1000//interval, dpi=150)
            print(f"Animation saved as {save_path_gif}")
            return
        print(f"Sequential animation saved successfully: {save_path}")
    except Exception as e:
        print(f"Failed to save animation: {e}")
        print("Try installing pillow (pip install pillow) for GIF support")
    
    plt.close(fig)