"""
simulation_analysis.py 

Code created by Lavya. 
Uses Chimera Reader (added to repository)
"""

import numpy as np
import reader
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.integrate import simpson
from scipy.fft import fft, fftfreq
from scipy.signal import savgol_filter

class Single_Sim():
    def __init__(self, simulation, ROOTDIR, project, start_step, final_step, step_interval, dump_freq,
                 burn_in=0, amp_threshold=1e-5, r2_threshold = 0.98, slice_dir="y", interface="x", integration="z",
                 vis_rho=True, vis_rho_step=200, breakout=False, breakout_threshold=0.03, cell_threshold=0,
                 streak=False, streak_domain=None, streak_dump=None,
                 streakbreakout=False, streakbreakout_domain=None):
        
        """
        Class: Single Sim. 
        This class is used to load data from a single simulation directory. It manages all the data loading and transformations. 
        It also comes equipped with a bunch of plotting functions to visualize and debug each simulation individually. 

        The class is extended between different files as an abc class. 

        NOTE: This function needs atleast one VTI file from which to load the grid. If not specified, this analysis code will not work. 
        NOTE: Breakout logic only works if target is set with its non-laser facing edge at x = 0. Otherwise this code will give garbage results. 
              This problem can be fixed easily in the data loader function in the breakout logic. This needs to be fixed separately for streaks and VTI data. 
        """

        #Load simulation and DAT file.
        self.simulation = simulation
        self.Chimera = reader.ChimeraSimulation(ROOTDIR, project)
        self.chimera_dat = self.Chimera.load_dat(self.simulation)

        #Creating timesteps from user specified input
        self.timesteps = np.arange(start_step, final_step, step_interval)
        self.dump_freq = dump_freq #Vti dump frequency

        #Configuring the streaks.
        self.streak = streak #This is the flag for if streak is used or not
        self.streak_domain = streak_domain #This is the streak file name
        self.streak_dump = streak_dump if streak_dump is not None else dump_freq #Streak frequency = Dump frequency if not specified otherwise.

        if self.streak and self.streak_domain is None:
            raise ValueError("streak=True requires streak_domain (Streak file name) to be set, e.g. 'streaks/xy0003Axial_Rho'.")

        #Flags for reading breakout time data from streak
        self.streakbreakout = streakbreakout #Breakout flag
        self.streakbreakout_domain = streakbreakout_domain #Breakout streak filename

        if self.streakbreakout and self.streakbreakout_domain is None:
            raise ValueError("streakbreakout=True requires streakbreakout_domain to be set.")

        """
        NOTE: This is not yet configured for a time file loaded as a DAT. That is a future modification!
        """
        if self.streak:
            self.times = self.timesteps * self.streak_dump #This math is because we will be using streaks only for data loading. 
        else:
            self.times = self.timesteps * dump_freq

        #Creating arrays for visualizing the densities.
        self.vis_rho_timesteps = np.arange(start_step, final_step, vis_rho_step)
        self.vis_rho = vis_rho

        #Burn in times and amplitude thresholds for linear region detection. 
        self.burn_in = burn_in #This is useful potentially for those physical regimes where the initial timestep is not yet influenced by laser/is acting weird in the FFT
        self.amp_threshold = amp_threshold 
        self.r2_threshold = r2_threshold

        #Directional slicing variables. These need to be correct wherever possible otherwise code will break.
        self.interface = interface
        self.slice = slice_dir
        self.integration = integration

        #Variables for breakout detection
        self.breakout = breakout
        self.breakout_threshold = breakout_threshold
        self.cell_threshold = cell_threshold #Tunable parameter to influence when breakout can be seen. Better to use VTI visualization to tune this. 
        self.breakout_densities = None
        self.breakout_pct_drop = None
        self.breakout_time = None

        #Loading the rho(t), coordinate data. Density is now a 2D function of x and t
        self.rho_t, self.xc, self.domain_size = self.data_loader()

        #Fourier transforming rho(t) to get the amplitudes (2D) and the frequences, wavelengths. Valid wavelengths allow for the removal of the zero mode. 
        self.a_x_t, self.freqs, self.valid_a_x_t, self.valid_freqs, self.wavelengths_m = \
            self.fourier_transformer(self.rho_t, self.xc)

        #Finding the linear regions by minimizing R^2 Error.
        self.growth_rates, self.start_indices, self.end_indices = \
            self.compute_linear_regions(self.times, self.valid_a_x_t, self.burn_in, self.amp_threshold)

        #Getting growth rates for the amplitude dominant mode (this is an important output for sweeprs)
        amp_dom_idx = np.argmax(np.max(self.valid_a_x_t, axis=0))
        self.dom_a = self.valid_a_x_t[:, amp_dom_idx]
        self.dom_a_growth = self.growth_rates[amp_dom_idx]
        self.dom_a_lambda = self.wavelengths_m[amp_dom_idx]
        
        #Safeguards
        if self.start_indices[amp_dom_idx] != -1 and self.end_indices[amp_dom_idx] < len(self.times):
            self.dom_a_linstart = self.times[self.start_indices[amp_dom_idx]]
            self.dom_a_linend = self.times[self.end_indices[amp_dom_idx]]
        else:
            self.dom_a_linstart = None
            self.dom_a_linend = None

        #Getting growth rates for the mode that grows the fastest (important outupt, but not as much as amplitude dominant mode)
        if np.all(np.isnan(self.growth_rates)):
            growth_dom_idx = 0
            self.dom_g = self.valid_a_x_t[:, growth_dom_idx]
            self.dom_g_growth = np.nan
            self.dom_g_lambda = self.wavelengths_m[growth_dom_idx]
            self.dom_g_linstart = None
            self.dom_g_linend = None
        else:
            growth_dom_idx = np.nanargmax(self.growth_rates)
            self.dom_g = self.valid_a_x_t[:, growth_dom_idx]
            self.dom_g_growth = self.growth_rates[growth_dom_idx]
            self.dom_g_lambda = self.wavelengths_m[growth_dom_idx]
            
            #Safeguards
            if self.start_indices[growth_dom_idx] != -1 and self.end_indices[growth_dom_idx] < len(self.times):
                self.dom_g_linstart = self.times[self.start_indices[growth_dom_idx]]
                self.dom_g_linend = self.times[self.end_indices[growth_dom_idx]]
            else:
                self.dom_g_linstart = None
                self.dom_g_linend = None

    #### PLOT: Density Plotting Function ####
    def density_plotter(self, Z_edges, X_edges, density_2d, X_centers, x_density, t, col_label="Z", row_label="X"):
        """
        Plots the spatial, 2D density, used for debugging. Also plots the axially integrated density to validate the integration.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        pcm = ax1.pcolormesh(Z_edges, X_edges, density_2d, norm=LogNorm(), cmap='turbo')
        cbar = fig.colorbar(pcm, ax=ax1)
        cbar.set_label("Density (rho_CH)", fontsize=12)
        ax1.set_title(f"Simulation {self.simulation}: Density Profile Slice at timestep = {t}", fontsize=14, fontweight='bold')
        ax1.set_xlabel(f"{col_label} Coordinate", fontsize=12)
        ax1.set_ylabel(f"{row_label} Coordinate", fontsize=12)
        ax1.set_xlim(Z_edges.min(), Z_edges.max())
        ax1.set_ylim(X_edges.min(), X_edges.max())

        ax2.plot(X_centers, x_density, color='#1f77b4', linewidth=2.5)
        ax2.fill_between(X_centers, x_density, color='#1f77b4', alpha=0.15)
        ax2.grid(True, linestyle='--', alpha=0.6, zorder=0)
        ax2.set_title(f"Mass Density Integrated over {self.integration}", fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel(f"{row_label} Coordinate", fontsize=12)
        ax2.set_ylabel("Integrated Density (rho_CH)", fontsize=12)
        ax2.set_xlim(X_centers.min(), X_centers.max())
        y_margin = (x_density.max() - x_density.min()) * 0.05
        ax2.set_ylim(x_density.min() - y_margin, x_density.max() + y_margin)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()

    #### CALCULATION: Coordinate Map Function ####
    def _resolve_coord_map(self, xc, yc, zc, xb, yb, zb):
        """
        Function to check if slicing, integration and interface directions check out. 
        """

        #Assigning a coordinate map (2D) of what to expect when the slice direction has been popped out.
        if self.slice == "x":
            coord_map = {"y": (yc, 0), "z": (zc, 1)}
        elif self.slice == "y":
            coord_map = {"x": (xc, 0), "z": (zc, 1)}
        elif self.slice == "z":
            coord_map = {"x": (xc, 0), "y": (yc, 1)}
        else:
            raise ValueError("Invalid slicing direction selected.") #Fall back

        #More fallbacks incase there is direction mismatch. 
        if self.integration not in coord_map:
            raise ValueError(f"integration='{self.integration}' is incompatible with slice_dir='{self.slice}'. Valid options here: {list(coord_map.keys())}")
        if self.interface not in coord_map:
            raise ValueError(f"interface='{self.interface}' is incompatible with slice_dir='{self.slice}'. Valid options here: {list(coord_map.keys())}")
        if self.interface == self.integration:
            raise ValueError("interface and integration must be different axes.")

        #The surviving coord is the interface direction because the integration direction coordinate gets integrated out and is not surviving anymore. 
        surviving_items = {k: v[0] for k, v in coord_map.items() if k != self.integration}
        this_surviving_coord = next(iter(surviving_items.values()))
        return coord_map, this_surviving_coord

    #### CALCULATION: Data Loader (CRITICAL) ####
    def data_loader(self):
        """
        Function to load data, either from streaks for from VTI file. 
        This function is also responsible for integration and breakout calculations while looping through timesteps. 

        NOTE: time dat file integration pending. Changes to this function will be very important. 
        """

        #Console outputs and loading the simulation.
        print(f"Loading data for simulation: {self.simulation} (Streak Mode: {self.streak})")
        self.Chimera.load_simulation(self.simulation, suffix="vti", geometry=0)

        #Getting the coordinate grid from a VTI file! 
        self.Chimera.import_timestep(self.timesteps[0], arr_names=["rho_CH"])
        xb, yb, zb = self.Chimera.x, self.Chimera.y, self.Chimera.z
        xc, yc, zc = self.Chimera.xc, self.Chimera.yc, self.Chimera.zc

        #Domain size calculations. Basically the length of the interface. 
        if self.interface == "x":
            domain_size = np.max(xc) - np.min(xc)
        elif self.interface == "y":
            domain_size = np.max(yc) - np.min(yc)
        elif self.interface == "z":
            domain_size = np.max(zc) - np.min(zc)

        #Get the integration direction from the coordinate map logic.
        coord_map, surviving_coord = self._resolve_coord_map(xc, yc, zc, xb, yb, zb)
        coord, axis = coord_map[self.integration]
        expected_len = surviving_coord.shape[0]

        #Initialize empty lists to loop through time. 
        rho_t = []
        breakout_densities = []

        #Streak mode: Direct time data loaded. Stored for future use.
        #The timestep loop is not made redundant for convenience sake and to interface better with VTI data. 
        if self.streak:
            streak_df = self.Chimera.load_dat(self.simulation, external_file=self.streak_domain)
            if self.streakbreakout:
                streakbreakout_df = self.Chimera.load_dat(self.simulation, external_file=self.streakbreakout_domain)

        #Loop through timesteps (not times!!) to extract data. 
        for t in self.timesteps:

            #Calculation for if streaks are to be loaded
            if self.streak:
                #Fall backs for streak not loading
                idx = int(t)
                if idx >= streak_df.shape[0]:
                    raise ValueError(f"Requested row index {idx} (timestep {t}) exceeds streak file bounds.")
                
                #Each row in the dataframe corresponds to one timestep. 
                axial_rho = streak_df.iloc[idx, :].to_numpy(dtype=float)

                #Fall backs for delimiter mistakes. 
                if len(axial_rho) == expected_len + 1 and np.isnan(axial_rho[-1]):
                    axial_rho = axial_rho[:-1]
                #Fall back if the first column is for times (this is useful for data integration).
                elif len(axial_rho) == expected_len + 1 and not np.isnan(axial_rho[-1]):
                    axial_rho = axial_rho[1:]

                #Getting breakout data from streaks. 
                if self.streakbreakout:

                    #Finding the first negative index (this only works when target is starting from x = 0)
                    negative_idx = np.where(coord < 0)[0]

                    #Shifting indices by a threshold value. Different threshold values can be thought of as different sensitivies.
                    idx0 = int(negative_idx[-self.cell_threshold])

                    #Pull rho(x) (integrated over y) at the specific timestep. 
                    b_row = streakbreakout_df.iloc[idx, :].to_numpy(dtype=float)
                    
                    #Fallbacks for time as the first column. 
                    if len(b_row) == expected_len + 1 and np.isnan(b_row[-1]):
                        b_row = b_row[:-1]
                    elif len(b_row) == expected_len + 1:
                        b_row = b_row[1:]

                    #Append the integrated density at the threshold point for checks later.
                    breakout_density = np.log(b_row[idx0])
                    breakout_densities.append(breakout_density)


            else:
                #Loading all VTI Data. 
                self.Chimera.import_timestep(t, arr_names=["rho_CH"])
                density = self.Chimera.arr["rho_CH"]

                #Same slicing and integration coordinate logic as the streak mode.
                #Pop out the slice axis.
                if self.slice == "x":
                    density_2d = density[0, :, :]
                elif self.slice == "y":
                    density_2d = density[:, 0, :]
                elif self.slice == "z":
                    density_2d = density[:, :, 0]

                #Integrate the density over the chosen axis. Different methods can be used for validation. 
                #Scipy simpson is O(delta x 4) so that is chosen.
                #NOTE: For validation with streaks it might be better to use a simple np.sum because thats the same integration method used in the gorgon source code. 
                axial_rho = simpson(density_2d, x = coord, axis = axis)

                #Breakout logic for VTI file.
                #Difference compared to streak is now we don't have an integration, we just look at a slice in x and take a mean of it. 
                #NOTE: For validation compared to streaks it might be worth integrating instead of taking a mean. 
                negative_idx = np.where(coord < 0)[0]
                idx0 = int(negative_idx[-self.cell_threshold])
                interface_slice = np.take(density_2d, idx0, axis=axis)
                breakout_density = np.mean(np.log(interface_slice))
                breakout_densities.append(breakout_density)

                #Density debugging, calls the density plotter function to show the density at specific timesteps. 
                if self.vis_rho and t in self.vis_rho_timesteps:
                    print(f"Visualizing data for timestep: {t} and simulation: {self.simulation}")
                    if self.slice == "x":
                        self.density_plotter(zb, yb, density_2d, surviving_coord, axial_rho, t, col_label="Z", row_label="Y")
                    elif self.slice == "y":
                        self.density_plotter(zb, xb, density_2d, surviving_coord, axial_rho, t, col_label="Z", row_label="X")
                    elif self.slice == "z":
                        self.density_plotter(yb, xb, density_2d, surviving_coord, axial_rho, t, col_label="Y", row_label="X")

            #Check shapes if they work or not. This is a fallback
            if axial_rho.shape[0] != expected_len:
                raise ValueError(f"Inconsistent grid at timestep {t}: axial_rho has length {axial_rho.shape[0]}, expected {expected_len}.")

            #Make NaN's zeros to prevent FFT crashes. 
            axial_rho = np.nan_to_num(axial_rho, nan=0.0)
            rho_t.append(axial_rho)

        #Convering the time list of breakout densities into an array.
        breakout_densities = np.array(breakout_densities)
        
        #If there are breakout densities (this is a fall back)
        if len(breakout_densities) > 0:
            #First timestep is the reference (you expect no matter in this area)
            breakout_reference = breakout_densities[0]

            #Find percentage drop compared to reference over all timesteps.
            pct_drop = (breakout_reference - breakout_densities) / breakout_reference

            #Find indices where the percentage drop is greater than the threshold (set by user).
            above_thresh = np.where(pct_drop > self.breakout_threshold)[0]

            #Assign self variables for later access.
            self.breakout_densities = breakout_densities
            self.breakout_pct_drop = pct_drop

            #Breakout time is the first time where the percentage drop is below threshold.
            if above_thresh.size > 0:
                self.breakout_time = self.times[above_thresh[0]]
            else:
                self.breakout_time = None
        else:
            self.breakout_densities = None
            self.breakout_pct_drop = None
            self.breakout_time = None

        #Console outputs for breakout logic.
        if self.breakout or self.streakbreakout:
            if self.breakout_time is not None:
                print(f"Simulation {self.simulation}: breakout at t = {self.breakout_time:.4e} s (pct drop {self.breakout_pct_drop[above_thresh[0]]:.4f})")
            elif self.breakout_pct_drop is not None:
                print(f"Simulation {self.simulation}: no breakout detected (max pct drop {self.breakout_pct_drop.max():.4f})")
            else:
                print(f"Simulation {self.simulation}: breakout tracking skipped (no breakout data provided).")

        return rho_t, surviving_coord, domain_size #Output the rho(t), interface coordinate and the domain size. 

    #### CALCULATION: Fourier Transformer (CRITICAL) ####
    def fourier_transformer(self, rho_t, xc):
        """
        Computes an FFT for the two dimensional, integrated density profile given by the data loader. 
        """
        print(f"Starting fourier analysis for simulation: {self.simulation}")

        #Initializing lists and frequencies.
        a_x_t_list = []
        freqs = np.empty((1,))
        
        #Helper variables for frequencies
        N = len(xc)
        dx = xc[1] - xc[0]

        #Get frequencies from fftfreq
        frequencies = fftfreq(N, dx)
        #Only concerned with positive frequencies, per mm.
        pos_freqs = frequencies[:N // 2] * 1e-3

        #Extracting frequencies and a(x,t)
        for rho in rho_t:
            if len(rho) != N:
                raise ValueError(
                    f"rho row has length {len(rho)} but xc has length {N}. "
                    f"These must match - check data_loader for inconsistent grids."
                )
            
            #Get the mess of complex numbers that the fourier transform returns.
            fourier_complex = fft(rho)
            amplitudes = (2.0 / N) * np.abs(fourier_complex)
            amplitudes[0] = amplitudes[0] / 2.0 #Normalize amplitudes.
            pos_amps = amplitudes[:N // 2]

            #Append amplitudes for different modes to the time list. 
            a_x_t_list.append(pos_amps)

        freqs = pos_freqs

        #Converting a(x,t) from a list to an array
        a_x_t = np.array(a_x_t_list)

        #Converting frequencies to wavelengths
        valid_freqs = freqs[1:]
        valid_a_x_t = a_x_t[:, 1:]

        #Changing units for the wavelength. 
        wavelengths_mm = 1.0 / valid_freqs
        wavelengths_m = wavelengths_mm * 1e-3

        return a_x_t, freqs, valid_a_x_t, valid_freqs, wavelengths_m

    #### CALCULATION: Find Linear Regions ####
    def find_anchored_linear_region(self, t_sliced, log_amp_sliced, min_window=10, r2_threshold=0.98):
        """
        Function computes the start and end indices of the linear growth region for each mode. 
        Maximizes R^2 to get the linear region. 
        Needs a minimum window for different line segments. 

        NOTE: This requires an amplitude threshold to clamp the first part so that the linear region only starts as the mode starts instead of loading
              up whenever the mode has a good straight curve (think fitting an oscillatory function). 
              This clamped function is passed in already in log scale. 
        """
        n_points = len(t_sliced)

        #Fall Back.
        if n_points <= min_window:
            return max(0, n_points - 1), 0

        #Initializing variables.
        best_length = 0
        best_r2 = -np.inf
        best_slope = 0
        best_end = n_points - 1  #Default fallback

        fallback_r2 = -np.inf
        fallback_slope = 0
        fallback_end = n_points - 1 

        start_idx = 0

        #Start indices from the minimum window possible and span the different lengths. 
        for end_idx in range(min_window, n_points):

            #Getting the slice arrays, corresponding to the index. 
            t_window = t_sliced[start_idx:end_idx]
            y_window = log_amp_sliced[start_idx:end_idx]

            #Get region length
            window_length = end_idx - start_idx

            #### CALCULATION FOR R^2 ####
            slope, intercept = np.polyfit(t_window, y_window, 1)

            if slope <= 0:
                continue

            y_pred = slope * t_window + intercept
            ss_res = np.sum((y_window - y_pred)**2)
            ss_tot = np.sum((y_window - np.mean(y_window))**2)

            if ss_tot == 0:
                continue

            r2 = 1 - (ss_res / ss_tot)

            if r2 > fallback_r2:
                fallback_r2 = r2
                fallback_slope = slope
                fallback_end = end_idx

            if r2 > r2_threshold:
                if window_length > best_length or (window_length == best_length and r2 > best_r2):
                    best_length = window_length
                    best_r2 = r2
                    best_slope = slope
                    best_end = end_idx
        
        #If the solver finds a linear region, show it otherwise give the fallback. 
        if best_length > 0:
            return best_end, best_slope
        else:
            return fallback_end, fallback_slope

    #### CALCULATION: Get Growth Rates and Linear Region Dependent Quantities ####
    def compute_linear_regions(self, t_array, amplitudes_2d, skip_steps=0, amp_threshold=1e-4, min_window=10):
        """
        Function that couples to the find_anchored_linear_region function and gives the growth rates, linear region start and end indices. 

        Loops through modes and finds the growth rate per mode. Uses savgol filter to smooth things out when necessary.
        """

        #Number of modes. 
        n_modes = amplitudes_2d.shape[1]
        
        #Initializing the arrays before looping through the modes.
        growth_rates = np.full(n_modes, np.nan)
        start_indices = np.full(n_modes, -1, dtype=int)
        end_indices = np.full(n_modes, -1, dtype=int)

        log_thresh = np.log(amp_threshold)
        t_post_burn = t_array[skip_steps:] #Add the burn in to the time arrays. 

        #Spanning (Looping through) each mode. 
        for i in range(n_modes):
            amp_post_burn = amplitudes_2d[skip_steps:, i] #Burn out the initial timesteps for amplitude.
            log_amp = np.log(amp_post_burn + 1e-12) #Convert the amplitude to log scale.

            #Finding the indices to where we want to clamp the linear region detector.
            gated_amp = np.where(log_amp > log_thresh, amp_post_burn, 0.0)
            nonzero_idx = np.where(gated_amp > 0.0)[0]

            #Fallback
            if len(nonzero_idx) == 0:
                continue

            thresh_idx = nonzero_idx[0]

            #Evaluation time with these thresholds.
            t_eval = t_post_burn[thresh_idx:]
            
            if len(t_eval) <= min_window:
                continue
            
            log_amp_eval = log_amp[thresh_idx:]

            #Savgol filter for the highly noisy modes. If this doesn't work the r^2 process can get messed up. 
            smooth_window = min(15, len(log_amp_eval))
            if smooth_window % 2 == 0:
                smooth_window -= 1

            if smooth_window > 3:
                smoothed_log_amp = savgol_filter(log_amp_eval, window_length=smooth_window, polyorder=2)
            else:
                smoothed_log_amp = log_amp_eval

            #Use the find anchored linear region function to get the data we need. 
            rel_end, slope = self.find_anchored_linear_region(t_eval, smoothed_log_amp, min_window=min_window, r2_threshold=self.r2_threshold)

            #Start needs to account for the burn in time.
            absolute_start = skip_steps + thresh_idx
            absolute_end = min(absolute_start + rel_end, len(t_array) - 1)

            growth_rates[i] = slope
            start_indices[i] = absolute_start
            end_indices[i] = absolute_end

        return growth_rates, start_indices, end_indices #Output all the useful data.

    #### PLOT: Show how the amplitude of each frequency grows with time (Useful to see a linear growth diverge into non-linear modes) ####
    def a_x_t_plot(self, save=False, show=True):
        plt.figure(figsize=(10, 6))

        threshold = 1e-2 * np.max(self.a_x_t)
        active_indices = np.where(np.max(self.a_x_t, axis=0) > threshold)[0]
        highest_active_index = active_indices[-1]

        pcm = plt.pcolormesh(self.freqs, self.times, self.a_x_t, shading="nearest", cmap="magma")
        cbar = plt.colorbar(pcm)
        cbar.set_label("Amplitude", fontsize=12)

        plt.title("Density Mode Evolution Over Time", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel(f"Spatial Frequency along {self.interface} (mm⁻¹)", fontsize=12)
        plt.ylabel("Time (Seconds)", fontsize=12)

        plt.xlim(self.freqs.min(), self.freqs[highest_active_index])
        plt.ylim(self.times.min(), self.times.max())
        plt.tight_layout()

        if show:
            plt.show()

    #### PLOT: Plots the linear region start and end times of different modes in the spectrum ####
    def timing_line_map_plotter(self, save=False, show=True, n_modes=20):
        plt.figure(figsize=(10, 6))

        amp_dom_idx = np.argmax(np.max(self.valid_a_x_t, axis=0))
        mode_step = max(1, int(self.wavelengths_m.size / (n_modes - 1)))
        bg_modes = np.arange(0, self.wavelengths_m.size, mode_step).tolist()

        if amp_dom_idx not in bg_modes:
            bg_modes[0] = amp_dom_idx

        modes_to_plot = sorted(list(set(bg_modes)))[:n_modes]
        colors = plt.cm.turbo(np.linspace(0, 1, len(modes_to_plot)))

        for i, m in enumerate(modes_to_plot):
            if self.start_indices[m] == -1 or self.end_indices[m] == -1:
                continue

            start_t = self.times[self.start_indices[m]]
            end_t = self.times[self.end_indices[m]]
            wave = self.wavelengths_m[m]

            if start_t == end_t:
                continue

            if m == amp_dom_idx:
                plt.plot([wave, wave], [start_t, end_t], color='red', linewidth=3.5, zorder=10,
                         marker='o', markersize=6, label=f"Max Amp Mode ({wave:.2e} m)")
            else:
                plt.plot([wave, wave], [start_t, end_t], color=colors[i], linewidth=1.5, alpha=0.6,
                         marker='o', markersize=4)

        plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
        plt.title("Linear Region Timeline (Start to Saturation)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel(rf"Wavelength, $\lambda$ along {self.interface} (m)", fontsize=12)
        plt.ylabel("Time (Seconds)", fontsize=12)

        plt.gca().invert_xaxis()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(loc='upper right')
        plt.tight_layout()

        if show:
            plt.show()

    #### PLOT: Plots Amplitude vs time for different modes ####
    def mode_plotter(self, save=False, show=True, n_modes=20):
        plt.figure(figsize=(10, 6))

        amp_dom_idx = np.argmax(np.max(self.valid_a_x_t, axis=0))
        growth_dom_idx = np.nanargmax(self.growth_rates)
        mode_step = int(self.wavelengths_m.size / (n_modes - 1))
        bg_modes = np.arange(0, self.wavelengths_m.size, mode_step)

        for m in bg_modes:
            if m != amp_dom_idx and m != growth_dom_idx:
                plt.plot(self.times[self.burn_in:], self.valid_a_x_t[self.burn_in:, m], label=rf"$\lambda$ = {self.wavelengths_m[m]:.2e} m", alpha=0.4)

        plt.plot(self.times[self.burn_in:], self.valid_a_x_t[self.burn_in:, amp_dom_idx], label=f"Max Amp = {self.wavelengths_m[amp_dom_idx]:.2e} m", color='red', linewidth=2.5, zorder=5)

        if amp_dom_idx != growth_dom_idx:
            plt.plot(self.times[self.burn_in:], self.valid_a_x_t[self.burn_in:, growth_dom_idx], label=f"Max Growth = {self.wavelengths_m[growth_dom_idx]:.2e} m", color='orange', linewidth=2.5, linestyle='--', zorder=5)

        plt.yscale('log')
        plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
        plt.title("Mode Amplitude Evolution", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Time (Seconds)", fontsize=12)
        plt.ylabel("Amplitude", fontsize=12)
        plt.xlim(self.times[self.burn_in:].min(), self.times[self.burn_in:].max())

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if show:
            plt.show()

    #### PLOT: Plots 20 modes with their linear regions to visualize region detection ####
    def grid_diagonistics(self, save=False, show=True, n_modes=20):
        amp_dom_idx = np.argmax(np.max(self.valid_a_x_t, axis=0))
        mode_step = max(1, int(self.wavelengths_m.size / (n_modes - 1)))
        bg_modes = np.arange(0, self.wavelengths_m.size, mode_step).tolist()

        if amp_dom_idx not in bg_modes:
            bg_modes[0] = amp_dom_idx

        modes_to_plot = sorted(list(set(bg_modes)))[:n_modes]
        cols = 5
        rows = int(np.ceil(len(modes_to_plot) / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
        axes = axes.flatten()

        for i, mode_idx in enumerate(modes_to_plot):
            ax = axes[i]
            dom_scale = self.wavelengths_m[mode_idx]
            t_post_burn = self.times[self.burn_in:]
            amp_post_burn = self.valid_a_x_t[self.burn_in:, mode_idx]

            absolute_start = self.start_indices[mode_idx]
            absolute_end = self.end_indices[mode_idx]
            ax.plot(t_post_burn, amp_post_burn, color='black', linewidth=1.5, alpha=0.7)

            no_region_found = (absolute_start == -1 or absolute_end == -1)

            if not no_region_found:
                t_linear = self.times[absolute_start:absolute_end]
                amp_linear = self.valid_a_x_t[absolute_start:absolute_end, mode_idx]
                if len(t_linear) > 1:
                    log_amp_linear = np.log(amp_linear + 1e-12)
                    slope, intercept = np.polyfit(t_linear, log_amp_linear, 1)
                    fit_line = np.exp(slope * t_linear + intercept)

                    ax.axvspan(self.times[absolute_start], self.times[max(absolute_end - 1, absolute_start)], color='#2ca02c', alpha=0.2)
                    ax.plot(t_linear, fit_line, color='red', linestyle='--', linewidth=2)

            ax.set_yscale('log')
            title_text = rf"$\lambda$ = {dom_scale:.2e} m"
            if no_region_found:
                title_text += "\n[never > threshold]"
            if mode_idx == amp_dom_idx:
                title_text += "\n[Max Amplitude Mode]"

            ax.set_title(title_text, fontsize=10, fontweight='bold' if mode_idx == amp_dom_idx else 'normal')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if i % cols == 0:
                ax.set_ylabel("Amplitude", fontsize=9)
            if i >= len(modes_to_plot) - cols:
                ax.set_xlabel("Time (s)", fontsize=9)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Automated Anchored Fits Across 20 Spectral Modes", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if show:
            plt.show()

    #### PLOT: Plots the dispersion relation (wavelength vs growth rates) ####
    def growth_rate_plotter(self, save=False, show=True):
        plt.figure(figsize=(20, 6))

        threshold = 1e-2 * np.max(self.valid_a_x_t)
        active_indices = np.where(np.max(self.valid_a_x_t, axis=0) > threshold)[0]
        highest_active_index = active_indices[-1]

        plt.plot(self.wavelengths_m, self.growth_rates, color='green', linewidth=2.5)
        plt.fill_between(self.wavelengths_m, self.growth_rates, color='green', alpha=0.15)
        plt.grid(True, linestyle='--', alpha=0.6, zorder=0)

        plt.title("Dispersion Relation (Growth Rate vs. Wavelength)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel(rf"Wavelength, $\lambda$ along {self.interface} (m)", fontsize=12)
        plt.ylabel(r"Growth Rate, $\gamma$ (s⁻¹)", fontsize=12)

        plt.xlim(self.wavelengths_m.max(), self.wavelengths_m[highest_active_index])
        
        # Check for NaN bounds safely
        if np.all(np.isnan(self.growth_rates)):
            pass
        else:
            y_margin = (np.nanmax(self.growth_rates) - np.nanmin(self.growth_rates)) * 0.05
            plt.ylim(np.nanmin(self.growth_rates) - y_margin, np.nanmax(self.growth_rates) + y_margin)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.gca().invert_xaxis()
        plt.tight_layout()

        if show:
            plt.show()