# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 14:12:47 2022

@author: add525
"""

#%% Libraries

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


#%% Functions

def setup_fontsize(size):
    '# Setup font sizes'
    
    if type(size) == int:
        
        mpl.rcParams['font.size'] = size
        
        mpl.rcParams["figure.titlesize"]=   'large' # default 'large'
        mpl.rcParams['axes.titlesize']  =   'large' # default 'large' = 'medium' * 1.2
        mpl.rcParams['axes.labelsize']  =   'medium' # default 'medium'
        mpl.rcParams['xtick.labelsize'] =   'small' # default 'medium'
        mpl.rcParams['ytick.labelsize'] =   'small' # default 'medium'
        mpl.rcParams['legend.fontsize'] =   'small' # default 'medium'
        
        return # print("Using {} fontsize.".format(size))
    
    elif size == "small":
        # XSmall = 12
        Small = 16
        Medium = 20
        Large = 24
        
    elif size == "large":
        # XSmall = 24
        Small = 32
        Medium = 40
        Large = 40
        
    mpl.rcParams['figure.titlesize']=   Large    
    mpl.rcParams['axes.titlesize']  =   Medium
    mpl.rcParams['axes.labelsize']  =   Medium
    mpl.rcParams['xtick.labelsize'] =   Small
    mpl.rcParams['ytick.labelsize'] =   Small
    mpl.rcParams['legend.fontsize'] =   Small
    
    mpl.rcParams["figure.dpi"] = 100
    
    return # print("Using {} fontsize.".format(size))



def plot_figure_axis(size="small", nplots=None, shape=None, ratio="square", output_ax=True):
    """
    Create a matplotlib Figure and Axes with flexible subplot layouts.
    
    Parameters
    ----------
    size : str or int
        Font/figure scaling. Passed to setup_fontsize().
    nplots : int, optional
        Creates a simple row of plots if shape not specified.
    shape : tuple or list
        Layout definition:
        - tuple (rows, cols): simple grid
        - nested list (e.g., [[1,1,2],[3,3,3]]): complex spanning layout
    ratio : "square" or (float, float)
        Controls the figure aspect ratio (width, height scaling).
    output_ax : bool
        If True, returns (fig, axes); otherwise, only fig.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : single axis or list of AxesSubplot
    """
    
    setup_fontsize(size)
    subplotlen = 6
    
    # --- Determine figure dimensions
    if ratio == "square":
        ratio = (1, 1)
    fig_width = subplotlen * ratio[0]
    fig_height = subplotlen * ratio[1]
    
    # --- Case 1: shape is provided (primary)
    if shape is not None:
        
        # --- Handle simple tuple shape (rows, cols)
        if isinstance(shape, tuple) and len(shape) == 2:
            rows, cols = shape
            fig, axes = plt.subplots(rows, cols, figsize=(fig_width * cols, fig_height * rows))
            axes = np.atleast_1d(axes).flatten()
            return (fig, axes[0]) if output_ax and len(axes) == 1 else (fig, axes)
        
        # --- Handle complex list-based shapes
        elif isinstance(shape, list):
            nrows = len(shape)
            ncols = max(len(row) for row in shape)
            fig = plt.figure(figsize=(fig_width * ncols, fig_height * nrows))
            gs = gridspec.GridSpec(nrows, ncols, figure=fig)
        
            axes_dict = {}
            for i, row in enumerate(shape):
                for j, cell in enumerate(row):
                    if cell == 0:  # allow placeholders
                        continue
                    if cell not in axes_dict:
                        axes_dict[cell] = [(i, j)]
                    else:
                        axes_dict[cell].append((i, j))
        
            axes = []
            for cell, positions in axes_dict.items():
                rows = [p[0] for p in positions]
                cols = [p[1] for p in positions]
                ax = fig.add_subplot(gs[min(rows):max(rows)+1, min(cols):max(cols)+1])
                axes.append(ax)
        
            return (fig, axes[0]) if output_ax and len(axes) == 1 else (fig, axes)
        
        else:
            raise ValueError("shape must be a tuple (rows, cols) or a nested list layout.")
            
    # --- Case 2: shape is None, use nplots for a single row
    elif nplots is not None:
        fig, axes = plt.subplots(1, nplots, figsize=(subplotlen*nplots*ratio[0], subplotlen*ratio[1]))
        axes = np.atleast_1d(axes).flatten()
    
    # --- Default fallback
    else:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        axes = [ax]

    return (fig, axes[0]) if output_ax and len(axes) == 1 else (fig, axes)


def plot(data, data_x=None, data_y=None, size="small", shape=(1, 1), ratio="square",
         plot_2D="pcolormesh", norm=None, cbar=False):
    """
    Generic single-axis plotting function.

    Works seamlessly with plot_figure_axis(). Designed for quick
    plotting of either 1D or 2D data on a single axis.

    Parameters
    ----------
    data : np.ndarray
        The data to plot. 1D → line plot, 2D → image plot.
    data_x, data_y : np.ndarray, optional
        Optional coordinate arrays for the data.
    size : str or int
        Font size setting (passed to setup_fontsize()).
    shape : tuple or list
        Layout for the subplots (passed to plot_figure_axis()).
        Only the first axis will be used for plotting.
    ratio : "square" or tuple(float,float)
        Aspect ratio scaling of the figure.
    plot_2D : "imshow" or "pcolormesh"
        Function to plot 2D data with.
    norm : "log" or None
        Use lognorm or regular image normalisation.
    cbar : bool
        Whether to include a colour bar.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The axis the data was plotted on.
    """
    
    # Create the figure and axes
    fig, axes = plot_figure_axis(size=size, shape=shape, ratio=ratio)
    
    # If multiple axes exist, take only the first one
    ax = axes if not isinstance(axes, (list, np.ndarray)) else axes[0]
    
    # ---- Plot the data ----
    if data.ndim == 1:
        # Handle 1D (line) data
        if data_x is None:
            data_x = np.arange(len(data))
        ax.plot(data_x, data)
    
    elif data.ndim == 2:
        # Handle 2D (image-like) data
        if norm == "log":
            from matplotlib.colors import LogNorm
            norm = LogNorm()
        else:
            norm = None
            if plot_2D == "imshow":
                if data_x is not None and data_y is not None:
                    im_extent = [min(data_x), max(data_x), min(data_y), max(data_y)]
                else:
                    im_extent = None
                im = ax.imshow(data, extent=im_extent, origin="lower", aspect="auto")
            elif plot_2D == "pcolormesh":
                im = ax.pcolormesh(data, norm=norm)
            else:
                raise Exception(f"Invalid plot method {plot_2D}.")
        if cbar:
            add_colourbar(fig, ax, im)
        
    elif data.ndim == 3 and data.shape[2] in [3, 4]:
        # Normalize if out of bounds
        if np.issubdtype(data.dtype, np.floating):
            if data.max() > 1 or data.min() < 0:
                data = (data - data.min()) / (data.max() - data.min())
        elif np.issubdtype(data.dtype, np.integer):
            if data.max() > 255:
                data = (255 * (data - data.min()) / (data.max() - data.min())).astype(np.uint8)
        ax.imshow(data)

    else:
        raise ValueError("data must be 1D or 2D")

    fig.tight_layout()
    return fig, ax


def add_legend(axis, label=None, variable="", species="", loc=None,
               line_include=None):
    '# Find what should be in the legend and add it'
    # If only one data set then no legend required

    # If label is not already provided
    if label is None:
        if isinstance(variable, list) and isinstance(species, list):
            label = [(spec,var) for spec in species for var in variable]
        elif isinstance(variable, list):
            label = [var for var in variable]
        elif isinstance(species, list):
            label = [spec for spec in species]
        else:
            label = [species + ", " + variable]
         
    # Modify label
    label = [s.replace("rhoNumber", "n") for s in label]
    label = [s.replace("Tsim", "T") for s in label]
    label = [s.replace("ion ", "") for s in label]
        
    # Enable legend if more than one label
    if len(label) > 1:
        # If there is only one axis all labels are associated with lines on that axis
        if not isinstance(axis, list):
            axis.legend(label)
        # If there are mulitple axis combine find all lines so labels can be combined
        else:
            lines = []
            lines.append(axis[0].get_lines())
            lines.append(axis[1].get_lines())
            lines = [item for sublist in lines for item in sublist]
            
            if line_include is not None:
                lines = [lines[i] for i in line_include]
            
            axis[-1].legend(lines, label, loc=loc)
    
    return


def add_colourbar(fig, axis, im, side="right", label=None, cbarpad=0.05, 
                  label_rot=90, labelpad=None):
    '''Adds a color bar to a given axis'''
    
    divider = make_axes_locatable(axis)
    cax = divider.append_axes(side, size='5%', pad=cbarpad)
    cbar = fig.colorbar(im, cax=cax)
    
    cbar.ax.yaxis.set_ticks_position(side)
    cbar.ax.yaxis.set_label_position(side)

    if label is not None:
        if labelpad is None:
            # Use matplotlib's default padding
            cbar.ax.set_ylabel(label, rotation=label_rot)
        else:
            # Use user-specified padding
            cbar.ax.set_ylabel(label, rotation=label_rot, labelpad=labelpad)

    return cbar


def add_colourbar_dual(fig, axis, im, side="right", label=None, cbarpad=0.01, size=0.03, label_rot=90, labelpad=None):
    """
    Adds a colorbar to an axis and allows precise control over its distance from the axis.

    Parameters:
        fig       : Matplotlib figure
        axis      : Main axis
        im        : The mappable image (pcolormesh, imshow, etc.)
        side      : 'left', 'right', 'top', 'bottom'
        cbarpad   : Fractional distance from the axis (0.0=touching, 1.0=far)
        size      : Fractional thickness of the colorbar
    """
    # Get axis position in figure coordinates
    pos = axis.get_position()
    
    if side in ["right", "left"]:
        cbar_width = size
        cbar_height = pos.height
        if side == "right":
            cbar_x = pos.x1 + cbarpad
        else:
            cbar_x = pos.x0 - cbarpad - cbar_width
        cbar_y = pos.y0
    elif side in ["top", "bottom"]:
        cbar_width = pos.width
        cbar_height = size
        if side == "top":
            cbar_y = pos.y1 + cbarpad
        else:
            cbar_y = pos.y0 - cbarpad - cbar_height
        cbar_x = pos.x0
    else:
        raise ValueError("side must be left, right, top, or bottom")

    # Create axes for the colorbar
    cax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
    cbar = fig.colorbar(im, cax=cax, orientation='vertical' if side in ["left","right"] else 'horizontal')

    # Put ticks and labels on the correct side
    if side == "left":
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
    if side == "right":
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('right')
    if side == "top":
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
    if side == "bottom":
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.ax.xaxis.set_label_position('bottom')

    if label is not None:
        cbar.ax.set_ylabel(label, rotation=label_rot, labelpad=labelpad)

    return cbar


# Interactive functions

def setup_crosshairs(fig, axs, show_text=False, lineout=True, lineout_h=0, lineout_v=0):
    """
    Add interactive crosshairs to one or more Matplotlib axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure containing the axes.
    axs : Axes or list of Axes
        One or more axes to attach crosshairs to.
    show_text : bool, optional
        Whether to show coordinates text in each axes.
    """
    
    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]

    crosshair_objects = []

    for ax in axs:
        if not ax.images:
            print(f"?? Skipping {ax}: no image found.")
            continue

        # --- Get image and data ---
        image = ax.images[0]
        data = image.get_array()
        ny, nx = data.shape
        
        bbox = ax.get_position()

        # --- Create small axes above for lineout ---
        lineout_height = 0.1 * bbox.height
        lineout_bottom = bbox.y1 + 0.01
        lineout_axh = fig.add_axes([
            bbox.x0, lineout_bottom,
            bbox.width, lineout_height
        ])
        
        # --- Create small axes to right for lineout ---
        lineout_width = 0.1 * bbox.width
        lineout_left = bbox.x1 + 0.01            # small gap to the right
        lineout_axv = fig.add_axes([
            lineout_left,       # x-position
            bbox.y0,            # same bottom as main axes
            lineout_width,      # width
            bbox.height         # same height as main axes
        ])

        # --- Initialize visuals ---
        vline = ax.axvline(color='white', alpha=0.8, lw=0.8, ls='--')
        hline = ax.axhline(color='white', alpha=0.8, lw=0.8, ls='--')

        xdata = np.arange(nx)
        lineout_lineh, = lineout_axh.plot(xdata, np.zeros_like(xdata), color='r')
        lineout_axh.set_xlim(0, nx)
        # lineout_axh.set_ylabel('Lineout')
        lineout_axh.set_xticks([])
        lineout_axh.set_yticks([])
        
        ydata = np.arange(ny)
        lineout_linev, = lineout_axv.plot(np.zeros_like(ydata), ydata, color='r')
        lineout_axv.set_ylim(0, ny)
        # lineout_axv.set_ylabel('Lineout')
        lineout_axv.set_xticks([])
        lineout_axv.set_yticks([])

        # Store state for this axes
        obj = {
            "ax": ax,
            "lineout_axh": lineout_axh,
            "lineout_axv": lineout_axv,
            "vline": vline,
            "hline": hline,
            "lineout_lineh": lineout_lineh,
            "lineout_linev": lineout_linev,
            "data": data,
            "locked": False,
        }
        crosshair_objects.append(obj)
    
    # --- Event handlers ---
    def mouse_move(event):
        for obj in crosshair_objects:
            ax = obj["ax"]
            if event.inaxes != ax or obj["locked"]:
                continue
            if event.xdata is None or event.ydata is None:
                continue

            x, y = event.xdata, event.ydata
            obj["vline"].set_xdata([x, x])
            obj["hline"].set_ydata([y, y])

            data = obj["data"]
            y_idx = int(round(y))
            if 0 <= y_idx < data.shape[0]:
                xdata = np.mean(data[y_idx-lineout_h:y_idx+lineout_h+1, :], axis=0)
                obj["lineout_lineh"].set_ydata(xdata)
                obj["lineout_axh"].set_ylim(np.nanmin(data), np.nanmax(data))
            x_idx = int(round(x))
            if 0 <= x_idx < data.shape[1]:
                ydata = np.mean(data[:, x_idx-lineout_v:x_idx-lineout_v+1], axis=1)[::-1]
                obj["lineout_linev"].set_xdata(ydata)
                obj["lineout_axv"].set_xlim(np.nanmin(data[:, x_idx]), np.nanmax(data[:, x_idx]))
            fig.canvas.draw_idle()

    def mouse_click(event):
        for obj in crosshair_objects:
            if event.inaxes == obj["ax"] and event.button == 1:
                obj["locked"] = not obj["locked"]
                # print(f"{'Locked' if obj['locked'] else 'Unlocked'} {obj['ax'].get_title() or 'axes'}")

    # Connect once (shared across all)
    fig.canvas.mpl_connect('motion_notify_event', mouse_move)
    fig.canvas.mpl_connect('button_press_event', mouse_click)


if __name__ == "__main__":

    # shape = (2,1)
    shape = [
    [1, 2],
    [1, 3]
    ]    

    fig, ax = plot_figure_axis(shape=shape, ratio=[1,0.5])
    
    setup_crosshairs(fig, ax)