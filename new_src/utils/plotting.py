"""
Plotting utilities for GAOT results visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union
import matplotlib.colors as mcolors


def plot_estimates(u_inp: Optional[np.ndarray], u_gtr: np.ndarray, u_prd: np.ndarray,
                  x_inp: np.ndarray, x_out: np.ndarray,
                  names: Optional[List[str]] = None, symmetric: Optional[List[bool]] = None,
                  figsize: tuple = (15, 5), dpi: int = 100) -> plt.Figure:
    """
    Plot input, ground truth, and prediction fields.
    
    Args:
        u_inp: Input field data [n_nodes, n_inp_channels] or None
        u_gtr: Ground truth field data [n_nodes, n_out_channels]
        u_prd: Predicted field data [n_nodes, n_out_channels]
        x_inp: Input coordinates [n_nodes, coord_dim]
        x_out: Output coordinates [n_nodes, coord_dim]
        names: Names for each channel
        symmetric: Whether each channel should use symmetric colormap
        figsize: Figure size
        dpi: Figure DPI
        
    Returns:
        plt.Figure: Generated figure
    """
    # Determine number of channels to plot
    n_out_channels = u_gtr.shape[-1]
    n_inp_channels = u_inp.shape[-1] if u_inp is not None else 0
    
    # Calculate subplot layout
    n_cols = 3 if u_inp is not None else 2  # Input (optional), GT, Pred
    n_rows = max(n_out_channels, n_inp_channels) if u_inp is not None else n_out_channels
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Set default names if not provided
    if names is None:
        names = [f'Channel_{i}' for i in range(max(n_out_channels, n_inp_channels))]
    
    # Set default symmetric flags if not provided
    if symmetric is None:
        symmetric = [True] * max(n_out_channels, n_inp_channels)
    
    # Plot input fields (if available)
    if u_inp is not None and n_cols >= 3:
        for i in range(n_inp_channels):
            ax = axes[i, 0] if n_rows > 1 else axes[0]
            _plot_field(ax, x_inp, u_inp[:, i], 
                       title=f'Input: {names[i] if i < len(names) else f"Channel_{i}"}',
                       symmetric=symmetric[i] if i < len(symmetric) else True)
        
        # Hide unused input subplots
        for i in range(n_inp_channels, n_rows):
            if n_rows > 1:
                axes[i, 0].set_visible(False)
    
    # Determine GT and Pred column indices
    gt_col = 1 if u_inp is not None else 0
    pred_col = 2 if u_inp is not None else 1
    
    # Plot ground truth and predictions
    for i in range(n_out_channels):
        # Ground truth
        ax_gt = axes[i, gt_col] if n_rows > 1 else axes[gt_col]
        _plot_field(ax_gt, x_out, u_gtr[:, i], 
                   title=f'Ground Truth: {names[i] if i < len(names) else f"Output_{i}"}',
                   symmetric=symmetric[i] if i < len(symmetric) else True)
        
        # Prediction
        ax_pred = axes[i, pred_col] if n_rows > 1 else axes[pred_col]
        _plot_field(ax_pred, x_out, u_prd[:, i], 
                   title=f'Prediction: {names[i] if i < len(names) else f"Output_{i}"}',
                   symmetric=symmetric[i] if i < len(symmetric) else True)
    
    # Hide unused output subplots
    for i in range(n_out_channels, n_rows):
        if n_rows > 1:
            axes[i, gt_col].set_visible(False)
            axes[i, pred_col].set_visible(False)
    
    plt.tight_layout()
    return fig


def _plot_field(ax: plt.Axes, coords: np.ndarray, values: np.ndarray, 
               title: str, symmetric: bool = True, cmap: str = None):
    """
    Plot a single field on given axes.
    
    Args:
        ax: Matplotlib axes
        coords: Coordinates [n_points, coord_dim]
        values: Field values [n_points]
        title: Plot title
        symmetric: Whether to use symmetric color scale
        cmap: Colormap name (optional)
    """
    if coords.shape[1] == 2:
        # 2D case
        x, y = coords[:, 0], coords[:, 1]
        
        # Choose colormap
        if cmap is None:
            cmap = 'RdBu_r' if symmetric else 'viridis'
        
        # Set color limits
        if symmetric:
            vmax = np.max(np.abs(values))
            vmin = -vmax
        else:
            vmin, vmax = np.min(values), np.max(values)
        
        # Create scatter plot
        scatter = ax.scatter(x, y, c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=1.0)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, shrink=0.8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
    
    elif coords.shape[1] == 1:
        # 1D case
        x = coords[:, 0]
        ax.plot(x, values, '-o', markersize=2)
        ax.set_xlabel('X')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    else:
        # 3D or higher - project to 2D for visualization
        x, y = coords[:, 0], coords[:, 1]
        
        if cmap is None:
            cmap = 'RdBu_r' if symmetric else 'viridis'
        
        if symmetric:
            vmax = np.max(np.abs(values))
            vmin = -vmax
        else:
            vmin, vmax = np.min(values), np.max(values)
        
        scatter = ax.scatter(x, y, c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=1.0)
        plt.colorbar(scatter, ax=ax, shrink=0.8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
    
    ax.set_title(title)


def plot_loss_curves(train_losses: List[float], val_losses: List[float] = None,
                     figsize: tuple = (10, 6), save_path: str = None) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        plt.Figure: Generated figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses is not None:
        val_epochs = range(1, len(val_losses) + 1)
        ax.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Use log scale if losses span multiple orders of magnitude
    if max(train_losses) / min(train_losses) > 100:
        ax.set_yscale('log')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_error_distribution(errors: np.ndarray, figsize: tuple = (10, 6),
                           save_path: str = None) -> plt.Figure:
    """
    Plot distribution of prediction errors.
    
    Args:
        errors: Array of error values
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        plt.Figure: Generated figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax1.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(errors, vert=True)
    ax2.set_ylabel('Error')
    ax2.set_title('Error Statistics')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {np.mean(errors):.4f}\n'
    stats_text += f'Std: {np.std(errors):.4f}\n'
    stats_text += f'Median: {np.median(errors):.4f}\n'
    stats_text += f'Max: {np.max(errors):.4f}'
    
    ax2.text(1.1, 0.5, stats_text, transform=ax2.transAxes, 
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_field_comparison(field1: np.ndarray, field2: np.ndarray, coords: np.ndarray,
                         titles: List[str] = None, figsize: tuple = (15, 5),
                         symmetric: bool = True) -> plt.Figure:
    """
    Compare two fields side by side.
    
    Args:
        field1: First field [n_points, n_channels]
        field2: Second field [n_points, n_channels]  
        coords: Coordinates [n_points, coord_dim]
        titles: Titles for the fields
        figsize: Figure size
        symmetric: Whether to use symmetric colorscale
        
    Returns:
        plt.Figure: Generated figure
    """
    if titles is None:
        titles = ['Field 1', 'Field 2']
    
    n_channels = field1.shape[1]
    fig, axes = plt.subplots(n_channels, 3, figsize=figsize)
    
    if n_channels == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_channels):
        # Plot field 1
        _plot_field(axes[i, 0], coords, field1[:, i], 
                   title=f'{titles[0]} - Ch{i}', symmetric=symmetric)
        
        # Plot field 2
        _plot_field(axes[i, 1], coords, field2[:, i], 
                   title=f'{titles[1]} - Ch{i}', symmetric=symmetric)
        
        # Plot difference
        diff = field2[:, i] - field1[:, i]
        _plot_field(axes[i, 2], coords, diff, 
                   title=f'Difference - Ch{i}', symmetric=True)
    
    plt.tight_layout()
    return fig


def create_animation(fields: List[np.ndarray], coords: np.ndarray, 
                    save_path: str, titles: List[str] = None, 
                    interval: int = 200) -> None:
    """
    Create animation from sequence of fields.
    
    Args:
        fields: List of field arrays [n_timesteps][n_points, n_channels]
        coords: Coordinates [n_points, coord_dim]
        save_path: Path to save animation
        titles: Titles for each timestep
        interval: Interval between frames in milliseconds
    """
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("Matplotlib animation not available")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Initialize plot with first field
    field0 = fields[0][:, 0]  # First channel of first timestep
    
    if coords.shape[1] == 2:
        x, y = coords[:, 0], coords[:, 1]
        scatter = ax.scatter(x, y, c=field0, cmap='RdBu_r')
        plt.colorbar(scatter, ax=ax)
    else:
        x = coords[:, 0]
        line, = ax.plot(x, field0)
    
    def animate(frame):
        field = fields[frame][:, 0]  # First channel
        
        if coords.shape[1] == 2:
            scatter.set_array(field)
        else:
            line.set_ydata(field)
        
        title = titles[frame] if titles else f'Frame {frame}'
        ax.set_title(title)
        
        return [scatter] if coords.shape[1] == 2 else [line]
    
    anim = FuncAnimation(fig, animate, frames=len(fields), 
                        interval=interval, blit=True, repeat=True)
    
    # Save animation
    if save_path.endswith('.gif'):
        anim.save(save_path, writer='pillow', fps=1000//interval)
    elif save_path.endswith('.mp4'):
        anim.save(save_path, writer='ffmpeg', fps=1000//interval)
    else:
        print(f"Unsupported animation format: {save_path}")
    
    plt.close(fig)