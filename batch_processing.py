"""
batch_processing.py

Code created by Lavya.
"""

import numpy as np
import matplotlib.pyplot as plt

####
#LINE SWEEPER CLASS
####

class line_sweeper():
    def __init__(self, simulator, simulations, parameters, parameter_name):
        """
        Class to sweep a 1-dimensional parameter space. 
        Outputs a line plot for amplitude dominant mode, growth dominant mode and breakout times. 
        """
        #Loading the simulator
        self.simulator = simulator

        #Loading data from simulations
        self.data_loader(simulations)

        #Loading parameters as a np array
        self.parameters = np.array(parameters)
        self.parameter_name = parameter_name

        #Logical Condition
        if np.size(self.parameters) != len(simulations):
            raise ValueError("Number of simulations does not correspond with number of parameters")
    
    #### CALCULATION: Data Loading Function (Loops through Simulations) ####
    def data_loader(self, simulations):
        """
        Function to loop over simulations and extract relevant data. 
        """
        # Initializing lists for amplitude dominant mode.
        self.dom_a = []
        self.dom_a_growth = []
        self.dom_a_linstart = []
        self.dom_a_linend = []

        #Initializing lists for growth dominant mode.
        self.dom_g = []
        self.dom_g_growth = []
        self.dom_g_linstart = []
        self.dom_g_linend = []

        #Initializing lists for the general values.
        self.a_x_t = []
        self.start_indices = []
        self.end_indices = []
        self.wavelengths = []
        self.times = []

        #Initializing lists for the breakout times.
        self.breakout_time = []

        #Loop over simulations to load relevant data.
        for simulation in simulations:
            print(f"Solving for simulation: {simulation}")
            simulator = self.simulator(simulation)
            self.dom_a.append(simulator.dom_a)
            self.dom_a_growth.append(simulator.dom_a_growth)
            self.dom_a_linstart.append(simulator.dom_a_linstart)
            self.dom_a_linend.append(simulator.dom_a_linend)
            self.dom_g.append(simulator.dom_g)
            self.dom_g_growth.append(simulator.dom_g_growth)
            self.dom_g_linstart.append(simulator.dom_g_linstart)
            self.dom_g_linend.append(simulator.dom_g_linend)
            self.a_x_t.append(simulator.valid_a_x_t)
            self.start_indices.append(simulator.start_indices)
            self.end_indices.append(simulator.end_indices)
            self.wavelengths.append(simulator.wavelengths_m)
            self.times.append(simulator.times)
            self.breakout_time.append(
                simulator.breakout_time if simulator.breakout_time is not None else np.nan
            )
        
        #Converting the 1D lists into arrays
        self.dom_a_growth = np.array(self.dom_a_growth)
        self.dom_a_linstart = np.array(self.dom_a_linstart)
        self.dom_a_linend = np.array(self.dom_a_linend)
        self.dom_g_growth = np.array(self.dom_g_growth)
        self.dom_g_linstart = np.array(self.dom_g_linstart)
        self.dom_g_linend = np.array(self.dom_g_linend)
        self.breakout_time = np.array(self.breakout_time)

    #### PLOT: Plots data for the amplitude dominant mode ####
    def dominant_mode(self, show=True, save=False):
        # Create a 1x3 grid (1 row, 3 columns)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # --- Helper function for consistent styling across subplots ---
        def style_subplot(ax, y_data, ylabel, title, color, marker):
            ax.plot(self.parameters, y_data, color=color, linewidth=2.5, marker=marker, markersize=8)
            ax.fill_between(self.parameters, y_data, color=color, alpha=0.15)
            
            ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
            ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_xlabel(self.parameter_name, fontsize=12) # Added to all plots
            
            # Dynamic margins to prevent clipping
            y_margin = (np.max(y_data) - np.min(y_data)) * 0.1
            if y_margin == 0: y_margin = abs(np.max(y_data)) * 0.1 if np.max(y_data) != 0 else 0.1
            ax.set_ylim(np.min(y_data) - y_margin, np.max(y_data) + y_margin)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # 1. Growth Rate
        style_subplot(axes[0], self.dom_a_growth, "Growth Rate, $\gamma$ (s⁻¹)", 
                      "1. Growth Rate", '#1f77b4', 'o')
        
        # 2. Linear Start Time
        style_subplot(axes[1], self.dom_a_linstart, "Time (s)", 
                      "2. Linear Region Start Time", '#2ca02c', 'o')
        
        # 3. Saturation (Linear End) Time
        style_subplot(axes[2], self.dom_a_linend, "Time (s)", 
                      "3. Saturation (Linear End) Time", '#d62728', 'o')

        fig.suptitle(f"Amplitude Dominant Mode Sensitivity to {self.parameter_name}", fontsize=16, fontweight='bold', y=1.05)
        
        plt.tight_layout()
        if show:
            plt.show()

    #### PLOT: Plots data for the growth dominant mode ####
    def growth_mode(self, show=True, save=False):
        # Create a 1x3 grid (1 row, 3 columns)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # --- Helper function for consistent styling across subplots ---
        def style_subplot(ax, y_data, ylabel, title, color, marker, linestyle):
            ax.plot(self.parameters, y_data, color=color, linewidth=2.5, marker=marker, markersize=8, linestyle=linestyle)
            ax.fill_between(self.parameters, y_data, color=color, alpha=0.15)
            
            ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
            ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_xlabel(self.parameter_name, fontsize=12) # Added to all plots
            
            y_margin = (np.max(y_data) - np.min(y_data)) * 0.1
            if y_margin == 0: y_margin = abs(np.max(y_data)) * 0.1 if np.max(y_data) != 0 else 0.1
            ax.set_ylim(np.min(y_data) - y_margin, np.max(y_data) + y_margin)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # 1. Growth Rate
        style_subplot(axes[0], self.dom_g_growth, "Growth Rate, $\gamma$ (s⁻¹)", 
                      "1. Growth Rate", 'orange', 's', '--')
        
        # 2. Linear Start Time 
        style_subplot(axes[1], self.dom_g_linstart, "Time (s)", 
                      "2. Linear Region Start Time", '#9467bd', 's', '--')
        
        # 3. Saturation (Linear End) Time 
        style_subplot(axes[2], self.dom_g_linend, "Time (s)", 
                      "3. Saturation (Linear End) Time", '#8c564b', 's', '--')

        fig.suptitle(f"Growth Dominant Mode Sensitivity to {self.parameter_name}", fontsize=16, fontweight='bold', y=1.05)
        
        plt.tight_layout()
        if show:
            plt.show()

    #### PLOT: Plots data for the breakout times ####
    def breakout_time_plot(self, show=True, save=False):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(self.parameters, self.breakout_time, color='#e377c2', linewidth=2.5, marker='o', markersize=8)
        ax.fill_between(self.parameters, self.breakout_time, color='#e377c2', alpha=0.15)

        ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
        ax.set_title("Breakout Time Sensitivity", fontsize=13, fontweight='bold', pad=10)
        ax.set_ylabel("Breakout Time (s)", fontsize=12)
        ax.set_xlabel(self.parameter_name, fontsize=12)

        if np.all(np.isnan(self.breakout_time)):
            print("Warning: no breakout detected for any simulation in this sweep - "
                  "skipping y-limit scaling.")
        else:
            y_margin = (np.nanmax(self.breakout_time) - np.nanmin(self.breakout_time)) * 0.1
            if y_margin == 0 or np.isnan(y_margin):
                ref = np.nanmax(self.breakout_time)
                y_margin = abs(ref) * 0.1 if ref else 0.1
            ax.set_ylim(np.nanmin(self.breakout_time) - y_margin, np.nanmax(self.breakout_time) + y_margin)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.suptitle(f"Breakout Time Sensitivity to {self.parameter_name}", fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()

        if save:
            fig.savefig("breakout_time_sweep.png", dpi=300, bbox_inches='tight')
        if show:
            plt.show()

####
#BRANE SWEEPER CLASS
####

class brane_sweeper():
    def __init__(self, simulator, simulations, parameters, parameter_names,
                 debug=False, debug_interval=100, debug_dir="debug_plots"):
        """
        Class to sweep a p-dimensional parameter space. 
        Gives data in the form of corner plots for amplitude dominant mode, growth dominant mode and shock breakout times. 
        """
        
        #Setup the simulator. Passed in as a lambda.
        self.simulator = simulator

        #Debug mode variables.
        self.debug = debug
        self.debug_interval = debug_interval
        self.debug_dir = debug_dir

        #Load simulation data using data loader function.
        self.data_loader(simulations)

        #Parameter lists.
        self.parameters = np.array(parameters)
        
        if self.parameters.ndim == 1:
            self.parameters = self.parameters.reshape(-1, 1)
            
        self.parameter_names = parameter_names #Make parameter names into a self variable.

        #Fallbacks for shape mismatches.
        if self.parameters.shape[0] != len(simulations):
            raise ValueError(f"Shape mismatch: {len(simulations)} simulations provided but {self.parameters.shape[0]} parameter points given.")
            
        if self.parameters.shape[1] != len(self.parameter_names):
            raise ValueError(f"Dimension mismatch: parameters array has {self.parameters.shape[1]} dimensions, but {len(self.parameter_names)} names were provided.")
    
    #### CALCULATION: Data Loading Function (Loops through Simulations) ####
    def data_loader(self, simulations):
        """
        Function to loop over simulations and extract relevant data. 
        """

        #Initializing lists for amplitude dominant mode.
        self.dom_a = []
        self.dom_a_growth = []
        self.dom_a_linstart = []
        self.dom_a_linend = []

        #Initializing lists for growth dominant mode.
        self.dom_g = []
        self.dom_g_growth = []
        self.dom_g_linstart = []
        self.dom_g_linend = []

        #Initializing lists for breakout times.
        self.breakout_time = []

        #Looping over simulations.
        for i, simulation in enumerate(simulations):
            print(f"Solving for simulation: {simulation}")
            simulator = self.simulator(simulation)
            
            self.dom_a.append(simulator.dom_a)
            self.dom_a_growth.append(simulator.dom_a_growth)
            self.dom_a_linstart.append(simulator.dom_a_linstart)
            self.dom_a_linend.append(simulator.dom_a_linend)
            
            self.dom_g.append(simulator.dom_g)
            self.dom_g_growth.append(simulator.dom_g_growth)
            self.dom_g_linstart.append(simulator.dom_g_linstart)
            self.dom_g_linend.append(simulator.dom_g_linend)

            self.breakout_time.append(
                simulator.breakout_time if simulator.breakout_time is not None else np.nan
            )

            #Debug mode control flows. Dump all diagnostics for chosen intervals.
            if self.debug and (i % self.debug_interval == 0):
                print(f"[debug] Saving single-sim diagnostics for simulation: {simulation}")
                simulator.mode_plotter(save=True, show=False, save_dir=self.debug_dir)
                simulator.timing_line_map_plotter(save=True, show=False, save_dir=self.debug_dir)
                simulator.grid_diagonistics(save=True, show=False, save_dir=self.debug_dir)
                simulator.growth_rate_plotter(save=True, show=False, save_dir=self.debug_dir)
        
        #Convert list to arrays for futher plotting, etc.
        self.dom_a_growth = np.array(self.dom_a_growth)
        self.dom_a_linstart = np.array(self.dom_a_linstart)
        self.dom_a_linend = np.array(self.dom_a_linend)
        self.dom_g_growth = np.array(self.dom_g_growth)
        self.dom_g_linstart = np.array(self.dom_g_linstart)
        self.dom_g_linend = np.array(self.dom_g_linend)
        self.breakout_time = np.array(self.breakout_time)

    #### PLOT: Draws a scatter matrix (helper function for corner plots) ####
    def _draw_scatter_matrix(self, data, labels, title, color):
        """Helper method to draw a triangular scatter matrix without MCMC constraints."""
        from matplotlib.ticker import MaxNLocator
        
        dim = data.shape[1]
        # Slightly tighter figure sizing
        fig, axes = plt.subplots(dim, dim, figsize=(2.2 * dim, 2.2 * dim))
        
        # Squeeze the subplots together to mimic a real corner plot
        plt.subplots_adjust(wspace=0.08, hspace=0.08)

        # When dim == 1, plt.subplots returns a bare Axes instead of a 2D
        # array, so np.atleast_2d keeps the indexing below uniform.
        axes = np.atleast_2d(axes)
        
        for i in range(dim):
            for j in range(dim):
                ax = axes[i, j]
                
                # Hide the upper triangle
                if i < j:
                    ax.axis('off')
                    continue
                    
                # Diagonal: Histograms
                elif i == j:
                    # Added edge color and dynamic bins for better visibility
                    bins = max(4, min(10, len(data)//2))
                    ax.hist(data[:, i], bins=bins, color=color, alpha=0.75, edgecolor='black', linewidth=1.2)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.set_yticks([]) # Histograms in corner plots usually drop the Y-axis counts
                
                # Lower triangle: Scatter plots
                else:
                    # Upgraded marker styling
                    ax.scatter(data[:, j], data[:, i], color=color, s=60, alpha=0.7, 
                               edgecolors='black', linewidths=0.8, zorder=5)
                    
                    ax.grid(True, linestyle=':', alpha=0.6, zorder=0)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    # Add margin padding so points don't touch the axes
                    ax.margins(x=0.15, y=0.15)
                
                # Tick management: Keep them sparse and clean to prevent overlap
                ax.xaxis.set_major_locator(MaxNLocator(4, prune='both'))
                ax.yaxis.set_major_locator(MaxNLocator(4, prune='both'))
                
                # Only show axis labels and ticks on the outer left/bottom edges
                if i == dim - 1:
                    ax.set_xlabel(labels[j], fontsize=11, fontweight='medium')
                    ax.tick_params(axis='x', rotation=45, labelsize=9)
                else:
                    ax.set_xticklabels([])
                    
                if j == 0 and i > 0:
                    ax.set_ylabel(labels[i], fontsize=11, fontweight='medium')
                    ax.tick_params(axis='y', labelsize=9)
                else:
                    if i != j: 
                        ax.set_yticklabels([])

        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        return fig
    
    #### PLOT: Plots data for the amplitude dominant mode ####
    def dominant_mode(self, show=True, save=False):
        # Added the original line_sweeper colors as the 4th item in the tuple
        metrics = [
            (self.dom_a_growth, r"Growth Rate $\gamma$ (s$^{-1}$)", "Growth Rate", '#1f77b4'),     # Blue
            (self.dom_a_linstart, "Start Time (s)", "Linear Start Time", '#2ca02c'),             # Green
            (self.dom_a_linend, "Saturation Time (s)", "Saturation Time", '#d62728')             # Red
        ]

        for metric_data, metric_label, title, plot_color in metrics:
            data_stacked = np.column_stack((self.parameters, metric_data))
            labels = self.parameter_names + [metric_label]
            
            fig = self._draw_scatter_matrix(
                data_stacked, 
                labels, 
                f"Amplitude Dominant Mode: {title}", 
                plot_color  # Passing the specific color here
            )
            
            if save:
                fig.savefig(f"amp_dom_{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
                
        if show:
            plt.show()

    #### PLOT: Plots data for the growth dominant mode ####
    def growth_mode(self, show=True, save=False):
        # Added the original line_sweeper colors here as well
        metrics = [
            (self.dom_g_growth, r"Growth Rate $\gamma$ (s$^{-1}$)", "Growth Rate", 'orange'),    # Orange
            (self.dom_g_linstart, "Start Time (s)", "Linear Start Time", '#9467bd'),             # Purple
            (self.dom_g_linend, "Saturation Time (s)", "Saturation Time", '#8c564b')             # Brown
        ]

        for metric_data, metric_label, title, plot_color in metrics:
            data_stacked = np.column_stack((self.parameters, metric_data))
            labels = self.parameter_names + [metric_label]
            
            fig = self._draw_scatter_matrix(
                data_stacked, 
                labels, 
                f"Growth Dominant Mode: {title}", 
                plot_color  # Passing the specific color here
            )
            
            if save:
                fig.savefig(f"growth_dom_{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
                
        if show:
            plt.show()

    #### PLOT: Plots data for the breakout times ####
    def breakout_time_plot(self, show=True, save=False):
        data_stacked = np.column_stack((self.parameters, self.breakout_time))
        labels = self.parameter_names + ["Breakout Time (s)"]

        fig = self._draw_scatter_matrix(
            data_stacked,
            labels,
            "Breakout Time",
            '#e377c2'
        )

        if save:
            fig.savefig("breakout_time.png", dpi=300, bbox_inches='tight')
        if show:
            plt.show()
