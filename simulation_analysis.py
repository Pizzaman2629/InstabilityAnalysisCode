""" simulation_analysis.py Code created by Lavya. Uses Chimera Reader (added to repository) """ 
import numpy as np 
import reader 
import os 
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm 
from scipy.integrate import simpson 
from scipy.fft import fft, fftfreq 
from scipy.signal import savgol_filter, find_peaks 

class Single_Sim(): 
    def __init__(self, simulation, ROOTDIR, project, start_step, final_step, step_interval, dump_freq, 
                 burn_in=0, min_window=10, r2_threshold = 0.98, 
                 slice_dir="y", interface="x", integration="z", 
                 vis_rho=True, vis_rho_step=200, 
                 breakout=False, breakout_threshold=0.03, cell_threshold=0, 
                 streak=False, streak_domain=None, streak_dump=None, streakbreakout=False, streakbreakout_domain=None, time_file = False): 
        """ Class: Single Sim. This class is used to load data from a single simulation directory. """ 
        #Loading simulation information. 
        self.simulation = simulation 
        self.Chimera = reader.ChimeraSimulation(ROOTDIR, project) 
        self.chimera_dat = self.Chimera.load_dat(self.simulation) 
        
        #Fallbacks incase dat file is not found. 
        if self.chimera_dat is None: 
            raise FileNotFoundError(f"Dat file not found for: {self.simulation}") 
            
        #Arranging timesteps for no timefile. 
        self.timesteps = np.arange(start_step, final_step, step_interval) 
        self.dump_freq = dump_freq 
        
        #Streak mode flags. 
        self.streak = streak 
        
        #Calculate directory and assign streak dump for streak mode. 
        if self.streak is True: 
            self.streak_domain = streak_domain + simulation + "Axial_Rho" 
            self.streak_dump = streak_dump if streak_dump is not None else dump_freq 
            
        #Fallbacks 
        if self.streak and self.streak_domain is None: 
            raise ValueError("streak=True requires streak_domain to be set.") 
            
        #Similar logic for the streak breakout. 
        self.streakbreakout = streakbreakout 
        if self.streakbreakout is True: 
            self.streakbreakout_domain = streakbreakout_domain + simulation + "Radial_Rho" 
            if self.streakbreakout and self.streakbreakout_domain is None: 
                raise ValueError("streakbreakout=True requires streakbreakout_domain to be set.") 
                
        #If timefile si not true, we use the input, interval steps and rescale accordingly. 
        if self.streak and time_file is not True: 
            self.times = self.timesteps * self.streak_dump 
        else: 
            self.times = self.timesteps * dump_freq 
            
        #Load in the timefile and round the indices if time file exists/is true. 
        if self.streak and time_file is True: 
            self.timedat = streak_domain + simulation + "Time" 
            self.times = self.Chimera.load_dat(self.simulation, external_file=self.timedat) 
            self.times = self.times.iloc[:, 0].to_numpy() 
            self.timesteps = np.round(self.times / self.streak_dump).astype(int) 
            self.timesteps[0] = 0 
            
        #Create timesteps to visualize the 2D density. 
        self.vis_rho_timesteps = np.arange(start_step, final_step, vis_rho_step) 
        self.vis_rho = vis_rho 
        
        #Assign helpful variables. 
        self.burn_in = burn_in 
        self.min_window = min_window 
        self.r2_threshold = r2_threshold 
        
        #Assign variables for integration, interface and slicing. 
        self.interface = interface 
        self.slice = slice_dir 
        self.integration = integration 
        
        #Assign breakout related variables. 
        self.breakout = breakout 
        self.breakout_threshold = breakout_threshold 
        self.cell_threshold = cell_threshold 
        self.breakout_densities = None 
        self.breakout_pct_drop = None 
        self.breakout_time = None 
        self.dx_uniform = None 
        
        #Load in axially integrated rho as a function of time, the interface grid and the domain size. 
        self.rho_t, self.xc, self.domain_size = self.data_loader() 
        
        #Fourier transform the input data and get amplitudes, frequencies and wavelengths in metres. 
        self.a_x_t, self.freqs, self.valid_a_x_t, self.valid_freqs, self.wavelengths_m = \
            self.fourier_transformer(self.rho_t, self.xc) 
            
        #Get growth rates using linear region detector 
        self.growth_rates, self.start_indices, self.end_indices = \
            self.compute_linear_regions(self.times, self.valid_a_x_t, self.burn_in, self.min_window) 
            
        #Amplitude dominant mode is the mode where the wavelength equals domain size (for the simulations we are concerned with, this can be changed later!) 
        target_lambda = self.domain_size / 2.0 
        self.amp_dom_idx = np.argmin(np.abs(self.wavelengths_m - target_lambda)) 
        
        #Get the data for amplitude dominant mode. If data doesn't exist, set to NaN value. 
        self.dom_a = self.valid_a_x_t[:, self.amp_dom_idx] 
        self.dom_a_lambda = self.wavelengths_m[self.amp_dom_idx] 
        if self.start_indices[self.amp_dom_idx] != -1 and self.end_indices[self.amp_dom_idx] < len(self.times): 
            self.dom_a_growth = self.growth_rates[self.amp_dom_idx] 
            self.dom_a_linstart = self.times[self.start_indices[self.amp_dom_idx]] 
            self.dom_a_linend = self.times[self.end_indices[self.amp_dom_idx]] 
        else: 
            # Mode found, but linear fit failed/absent - set properties to NaN/None 
            self.dom_a_growth = np.nan 
            self.dom_a_linstart = None 
            self.dom_a_linend = None 
            
        #Similarly, get the data for the max growth mode, if data doesn't exist, set to NaN value. 
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
            if self.start_indices[growth_dom_idx] != -1 and self.end_indices[growth_dom_idx] < len(self.times): 
                self.dom_g_linstart = self.times[self.start_indices[growth_dom_idx]] 
                self.dom_g_linend = self.times[self.end_indices[growth_dom_idx]] 
            else: 
                self.dom_g_linstart = None 
                self.dom_g_linend = None 

    #### PLOT: Visualize 2D, raw density #### 
    def density_plotter(self, Z_edges, X_edges, density_2d, X_centers, x_density, t, col_label="Z", row_label="X"): 
        """
        Function to plot the raw 2D density.
        Alongside the density, it plots the integrated density mode for visualization.
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

    #### CALCULATION: Coordinate Map #### 
    def _resolve_coord_map(self, xc, yc, zc, xb, yb, zb): 
        """ Contains all the raw information about which coordinate pairings are valid and which are not. """ 
        if self.slice == "x": 
            coord_map = {"y": (yc, 0), "z": (zc, 1)} 
        elif self.slice == "y": 
            coord_map = {"x": (xc, 0), "z": (zc, 1)} 
        elif self.slice == "z": 
            coord_map = {"x": (xc, 0), "y": (yc, 1)} 
        else: 
            raise ValueError("Invalid slicing direction selected.") 
            
        #Fallbacks incase the coordinate configuration chosen is invalid (can happen a lot!) 
        if self.integration not in coord_map: 
            raise ValueError(f"integration='{self.integration}' is incompatible with slice_dir='{self.slice}'.") 
        if self.interface not in coord_map: 
            raise ValueError(f"interface='{self.interface}' is incompatible with slice_dir='{self.slice}'.") 
        if self.interface == self.integration: 
            raise ValueError("interface and integration must be different axes.") 
            
        #Coordinate map returns appropriate interface and integration coordinates and removes the slice coordinates. 
        surviving_items = {k: v[0] for k, v in coord_map.items() if k != self.integration} 
        this_surviving_coord = next(iter(surviving_items.values())) 
        return coord_map, this_surviving_coord 

    #### CRITICAL: Data Loader #### 
    def data_loader(self): 
        """ Function to load data from Chimera. In VTI mode, loads raw density data, integrates over an axis and gives the axially integrated data. In Streak mode, loads axially integrated data directly from streak. Function also does breakout calculations by integrating over the interface and checking for a threhsold change. In streak mode, the integration is already obtained. NOTE: For most of the relevant simulations, the sim itself is terminated on breakout so a lot of work has been put in to make sure that the plottings work with NaN values because thats what most of the breakouts will be (none). """ 
        #Console output. 
        print(f"Loading data for simulation: {self.simulation} (Streak Mode: {self.streak})") 
        
        #Load simulation with a "vts" file. 
        self.Chimera.load_simulation(self.simulation, suffix="vts", geometry=0) 
        
        #Load the grid and the timestep function. 
        self.Chimera.import_timestep(self.timesteps[0], arr_names=["rho_CH"]) 
        xb, yb, zb = self.Chimera.x, self.Chimera.y, self.Chimera.z 
        xc, yc, zc = self.Chimera.xc, self.Chimera.yc, self.Chimera.zc 
        
        #Get the interface size. 
        if self.interface == "x": 
            domain_size = np.max(xc) - np.min(xc) 
        elif self.interface == "y": 
            domain_size = np.max(yc) - np.min(yc) 
        elif self.interface == "z": 
            domain_size = np.max(zc) - np.min(zc) 
            
        #Get the coordinate map and decide the integration axis. 
        coord_map, surviving_coord = self._resolve_coord_map(xc, yc, zc, xb, yb, zb) 
        coord, axis = coord_map[self.integration] 
        expected_len = surviving_coord.shape[0] #Expected length is the grid size for axial rho later.
        
        #Go 1 micron to the sides of x = 0 to get the dx value. 
        #NOTE: This works for a feathered grid which is uniform near x = 0, otherwise it does NOT work and should be changed! 
        idx0_init = np.argmin(np.abs(surviving_coord - (-1.0e-6))) 
        if idx0_init < len(surviving_coord) - 1: 
            self.dx_uniform = np.abs(surviving_coord[idx0_init + 1] - surviving_coord[idx0_init]) 
        else: 
            self.dx_uniform = np.abs(surviving_coord[idx0_init] - surviving_coord[idx0_init - 1]) 
            
        #Initialize empty lists for data loading. 
        rho_t = [] 
        breakout_densities = [] 

        #Streak mode logic.
        if self.streak: 
            #Load the streak file as a dataframe.
            streak_df = self.Chimera.load_dat(self.simulation, external_file=self.streak_domain) 

            #Get a maximum timestep (this logic is because sometimes the time file and streak file indices differ by just 1 which trips things up)
            max_valid_idx = min(len(self.timesteps), len(streak_df)) 
            self.timesteps = self.timesteps[:max_valid_idx] 
            self.times = self.times[:max_valid_idx] 

            #For breakouts, load the appropriate dat file.
            if self.streakbreakout: 
                streakbreakout_df = self.Chimera.load_dat(self.simulation, external_file=self.streakbreakout_domain) 

        #Loop over timesteps
        for idx, t in enumerate(self.timesteps): 

            #Streak mode logic.
            if self.streak: 
                if idx >= streak_df.shape[0]: 
                    raise ValueError(f"Requested row index {idx} exceeds streak file bounds.") 

                #Axial rho at the specific time (dataframe row)
                axial_rho = streak_df.iloc[idx, :].to_numpy(dtype=float)

                #Fallbacks to ensure axial rho matches grid sizes. 
                if len(axial_rho) == expected_len + 1 and np.isnan(axial_rho[-1]): 
                    axial_rho = axial_rho[:-1] 
                elif len(axial_rho) == expected_len + 1 and not np.isnan(axial_rho[-1]): 
                    axial_rho = axial_rho[1:] 

                #Streak breakout logic (redundant for when the gorgon sim terminates at breakout)
                if self.streakbreakout: 
                    negative_idx = np.where(coord < 0)[0] 
                    idx0 = np.argmin(np.abs(coord - (-1.0e-6))) 
                    b_row = streakbreakout_df.iloc[idx, :].to_numpy(dtype=float) 
                    if len(b_row) == expected_len + 1 and np.isnan(b_row[-1]): 
                        b_row = b_row[:-1] 
                    elif len(b_row) == expected_len + 1: 
                        b_row = b_row[1:] 
                    breakout_density = np.log(b_row[idx0]) 
                    breakout_densities.append(breakout_density) 

            #VTI/VTS file loading logic.
            else: 
                self.Chimera.import_timestep(t, arr_names=["rho_CH"]) 
                density = self.Chimera.arr["rho_CH"] #Load the density

                #Slice along the given direction.
                if self.slice == "x": 
                    density_2d = density[0, :, :] 
                elif self.slice == "y": 
                    density_2d = density[:, 0, :] 
                elif self.slice == "z": 
                    density_2d = density[:, :, 0] 

                #Integrate over the axis pulled from the coordinate map.
                axial_rho = simpson(density_2d, x = coord, axis = axis) 

                #Find the index 1 micron away from the interface (zero)
                negative_idx = np.where(coord < 0)[0] 
                idx0 = np.argmin(np.abs(coord - (-1.0e-6))) #The weird sign magic here keeps it negative.
                interface_slice = np.take(density_2d, idx0, axis=axis) 
                breakout_density = np.mean(np.log(interface_slice)) #Get mean value at that point and track it for reference. 
                breakout_densities.append(breakout_density) 

                #Plot density at this timestep if the timestep is destined for plotting.
                if self.vis_rho and t in self.vis_rho_timesteps: 
                    if self.slice == "x": 
                        self.density_plotter(zb, yb, density_2d, surviving_coord, axial_rho, t, col_label="Z", row_label="Y") 
                    elif self.slice == "y": 
                        self.density_plotter(zb, xb, density_2d, surviving_coord, axial_rho, t, col_label="Z", row_label="X") 
                    elif self.slice == "z": 
                        self.density_plotter(yb, xb, density_2d, surviving_coord, axial_rho, t, col_label="Y", row_label="X") 

            #Fallbacks incase there is a mismatch in sizes.          
            if axial_rho.shape[0] != expected_len: 
                raise ValueError(f"Inconsistent grid at timestep {t}: axial_rho has length {axial_rho.shape[0]}, expected {expected_len}.") 

            #Set the NaN values which can arise from fortran to zero.
            axial_rho = np.nan_to_num(axial_rho, nan=0.0) 

            #Append to the list.
            rho_t.append(axial_rho) 

        #Make the breakout densities list into array for better manipulation.    
        breakout_densities = np.array(breakout_densities) 

        #Breakout logic (pretty straightforward tbh)
        if len(breakout_densities) > 0: 
            breakout_reference = breakout_densities[0] 
            pct_drop = (breakout_reference - breakout_densities) / breakout_reference 
            above_thresh = np.where(pct_drop > self.breakout_threshold)[0] 
            self.breakout_densities = breakout_densities 
            self.breakout_pct_drop = pct_drop 
            if above_thresh.size > 0: 
                self.breakout_time = self.times[above_thresh[0]] 
            else: 
                self.breakout_time = None 

        #If no breakout is detected, set to None
        else: 
            self.breakout_densities = None 
            self.breakout_pct_drop = None 
            self.breakout_time = None 
            
        return rho_t, surviving_coord, domain_size #Return all important values 

    #### CRITICAL: FFT ####
    def fourier_transformer(self, rho_t, spatial_coord): 
        """
        Function to compute FFT. 
        
        NOTE: Currently this works for a uniform grid and a feathered grid provided the feathered grid is approximately uniform
            in all areas of interest. This will NOT work for a truly non-uniform grid in which case its better to switch the FFT function.
        """ 
        N = len(spatial_coord) 
        #Get resolution (this is setup to y if interface coordinate is y).
        dy = np.abs(spatial_coord[1] - spatial_coord[0]) 

        #Get the spatial frequencies and extract the positive ones.
        frequencies = fftfreq(N, dy) 
        pos_freqs = frequencies[:N // 2] 
        a_x_t_list = [] #Initialize empty amplitude list.

        #Loop over timesteps, compute FFT, get amplitudes, append.
        for rho in rho_t: 
            if len(rho) != N: 
                raise ValueError(f"Density array length mismatch. Expected {N}, got {len(rho)}.") 
            fourier_complex = fft(rho) 
            amplitudes = (2.0 / N) * np.abs(fourier_complex) 
            amplitudes[0] = amplitudes[0] / 2.0  #Normalize.
            pos_amps = amplitudes[:N // 2] 
            a_x_t_list.append(pos_amps) 

        #Convert list to array for happiness.
        a_x_t = np.array(a_x_t_list) 

        #Drop the zero mode so it doesn't muddle the calculations.
        valid_freqs = pos_freqs[1:] 
        valid_a_x_t = a_x_t[:, 1:] 

        #Convert frequencies to desired spatial coordinates (metres).
        wavelengths_m = 1.0 / valid_freqs 

        return a_x_t, pos_freqs, valid_a_x_t, valid_freqs, wavelengths_m #Return useful values.

    #### CALCULATION: Linear Region Detector ####
    def find_anchored_linear_region(self, t_sliced, log_amp_sliced, min_window=10, r2_threshold=0.98): 
        """
        Function to calculate the linear region for a singular mode. 

        NOTE: This is probably THE most iffy function, so when doing analysis make sure to optimize it nicely.

        NOTE: There is also a chance the second order derivative method may break for some fits, which is why
            it is good practice to always visualize the fits with the plotting tools provided in the class.
        """

        #Get the number of tiem points.
        n_points = len(t_sliced) 

        #If points are less than minimum window points, give a nan result (fallbacks)
        if n_points <= min_window: 
            return 0, max(0, n_points - 1), np.nan 

        #Get second order derivative of mode.
        dy = np.gradient(log_amp_sliced) 
        d2y = np.gradient(dy) 

        #Find the maximum second order derivative point.
        max_d2y_idx = np.argmax(np.abs(d2y))

        #Step 3 indices away from this to get a safe start position (since sometimes modes can have an insane cliff when being initialized!)
        safe_start = max_d2y_idx + 3
        safe_start = min(safe_start, n_points - min_window - 1)
        safe_start = max(0, safe_start)

        #Get maximum slope after trimming to the new start position.
        if safe_start < len(dy):
            max_slope_after = np.max(dy[safe_start:])
        else:
            max_slope_after = 0

        #Fallbacks
        if max_slope_after <= 0:
            return 0, max(0, n_points - 1), np.nan

        #Get a threshold to identify if the curve is flat or not.
        flat_threshold = 0.05 * max_slope_after

        #Step across indices to make sure we don't start at a flat point.
        #Clamp to the first index where curve is not flat!
        start_idx = safe_start
        while start_idx < n_points - min_window:
            if dy[start_idx] >= flat_threshold:
                break
            start_idx += 1
            
        if start_idx >= n_points - min_window:
            start_idx = max(0, n_points - min_window - 1)

        #R^2 error logic across windows.
        best_length = 0 
        best_r2 = -np.inf 
        best_slope = np.nan 
        best_end = n_points - 1 
        
        fallback_r2 = -np.inf 
        fallback_slope = np.nan 
        fallback_end = n_points - 1 
        
        for end_idx in range(start_idx + min_window, n_points): 
            t_window = t_sliced[start_idx:end_idx] 
            y_window = log_amp_sliced[start_idx:end_idx] 
            window_length = end_idx - start_idx 
            
            slope, intercept = np.polyfit(t_window, y_window, 1) 
            
            if slope <= 0 or slope * (t_window[-1] - t_window[0]) < 0.1: 
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

        #Return end index, start index and slope (growth rate)
        if best_length > 0: 
            return start_idx, best_end, best_slope 
        else: 
            return start_idx, fallback_end, fallback_slope 

    #### CALCULATION: Compute Growth Rates ####
    def compute_linear_regions(self, t_array, amplitudes_2d, skip_steps=0, min_window=2): 
        """
        Function to take in fourier transformed data and compute the growth rates for it.

        Interfaces with find_anchored_linear_region to compute linear regions for multiple modes.  
        """

        #Get the number of modes. 
        n_modes = amplitudes_2d.shape[1]

        #Initialize empty arrays for growth rates, start indices and end indices. 
        growth_rates = np.full(n_modes, np.nan) 
        start_indices = np.full(n_modes, -1, dtype=int) 
        end_indices = np.full(n_modes, -1, dtype=int) 

        #If we want burn, apply that.
        t_post_burn = t_array[skip_steps:] 

        #Loop over all modes.
        for i in range(n_modes): 
            #Slice amplitudes based on burn.
            amp_post_burn = amplitudes_2d[skip_steps:, i] 
            log_amp = np.log(amp_post_burn + 1e-12) #Switch to log scale. 

            #Fallbacks
            t_eval = t_post_burn 
            if len(t_eval) <= min_window: 
                continue 

            #Filter using savgol. Minimum filter window has to always be equal to the linear regions' minimum window.    
            log_amp_eval = log_amp 
            smooth_window = min(min_window, len(log_amp_eval)) 
            if smooth_window % 2 == 0: 
                smooth_window -= 1 
            if smooth_window > 3: 
                smoothed_log_amp = savgol_filter(log_amp_eval, window_length=smooth_window, polyorder=2) 
            else: 
                smoothed_log_amp = log_amp_eval 

            #Cap amplitude end where breakout occurs.    
            cap_idx = len(t_eval) - 1
            if (self.breakout or self.streakbreakout) and self.breakout_time is not None: 
                breakout_candidates = np.where(t_eval > self.breakout_time)[0] 
                if breakout_candidates.size > 0: 
                    cap_idx = breakout_candidates[0] 

            #Final evaluation timesteps, amplitudes. after start and end caps.        
            t_eval = t_eval[:cap_idx + 1] 
            smoothed_log_amp = smoothed_log_amp[:cap_idx + 1] 

            #Get the linear region.
            rel_start, rel_end, slope = self.find_anchored_linear_region( 
                t_eval, smoothed_log_amp, min_window=min_window, r2_threshold=self.r2_threshold 
            ) 
            
            if np.isnan(slope): 
                continue 

            #Append data.
            absolute_start = skip_steps + rel_start 
            absolute_end = min(skip_steps + rel_end, len(t_array) - 1) 
            growth_rates[i] = slope 
            start_indices[i] = absolute_start 
            end_indices[i] = absolute_end 
            
        return growth_rates, start_indices, end_indices #Return useful values.

    #### PLOT: Plot Various Heatmap of Mode Evolution ####
    def a_x_t_plot(self, save=False, show=True, save_dir="debug_plots"): 
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
        if save: 
            os.makedirs(save_dir, exist_ok=True) 
            plt.savefig(os.path.join(save_dir, f"{self.simulation}_a_x_t_plot.png"), dpi=300, bbox_inches='tight') 
        if show: 
            plt.show() 
        else: 
            plt.close() 

    #### PLOT: Plot Start and Stop Times of Various Modes ####
    def timing_line_map_plotter(self, save=False, show=True, n_modes=20, save_dir="debug_plots"): 
        plt.figure(figsize=(10, 6)) 
        # Replaced finding logic with inherited index 
        amp_dom_idx = self.amp_dom_idx 
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
                plt.plot([wave, wave], [start_t, end_t], color='red', linewidth=3.5, zorder=10, marker='o', markersize=6, label=f"Max Amp Mode ({wave:.2e} m)") 
            else: 
                plt.plot([wave, wave], [start_t, end_t], color=colors[i], linewidth=1.5, alpha=0.6, marker='o', markersize=4) 
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
        if save: 
            os.makedirs(save_dir, exist_ok=True) 
            plt.savefig(os.path.join(save_dir, f"{self.simulation}_timing_line_map.png"), dpi=300, bbox_inches='tight') 
        if show: 
            plt.show() 
        else: 
            plt.close() 

    #### PLOT: Plot the A vs T Curve for Various Modes ####
    def mode_plotter(self, save=False, show=True, n_modes=20, save_dir="debug_plots"): 
        plt.figure(figsize=(10, 6)) 
        # Replaced finding logic with inherited index 
        amp_dom_idx = self.amp_dom_idx 
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
        if save: 
            os.makedirs(save_dir, exist_ok=True) 
            plt.savefig(os.path.join(save_dir, f"{self.simulation}_mode_plot.png"), dpi=300, bbox_inches='tight') 
        if show: 
            plt.show() 
        else: 
            plt.close() 

    #### PLOT: Plot grid diagnostics for fit visualization ####
    def grid_diagonistics(self, save=False, show=True, n_modes=20, save_dir="debug_plots"): 
        # Replaced finding logic with inherited index 
        amp_dom_idx = self.amp_dom_idx 
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
        if save: 
            os.makedirs(save_dir, exist_ok=True) 
            plt.savefig(os.path.join(save_dir, f"{self.simulation}_grid_diagnostics.png"), dpi=300, bbox_inches='tight') 
        if show: 
            plt.show() 
        else: 
            plt.close() 

    #### PLOT: Plot Dispersion Relation ####
    def growth_rate_plotter(self, save=False, show=True, save_dir="debug_plots"): 
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
        if save: 
            os.makedirs(save_dir, exist_ok=True) 
            plt.savefig(os.path.join(save_dir, f"{self.simulation}_growth_rate.png"), dpi=300, bbox_inches='tight') 
        if show: 
            plt.show() 
        else: 
            plt.close() 

    #### PLOT: Plot the Dominant Mode and Its Linear Fit ####
    def dominant_mode_fit_plotter(self, save=False, show=True, save_dir="debug_plots"): 
        """ Plots the dominant amplitude mode and its linear fit. Left tile: Full time evolution. Right tile: Zoomed in view of the linear fit region with data point markers. """ 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6)) 
        amp_dom_idx = self.amp_dom_idx 
        dom_scale = self.wavelengths_m[amp_dom_idx] 
        # Extract data post burn-in 
        t_post_burn = self.times[self.burn_in:] 
        amp_post_burn = self.valid_a_x_t[self.burn_in:, amp_dom_idx] 
        absolute_start = self.start_indices[amp_dom_idx] 
        absolute_end = self.end_indices[amp_dom_idx] 
        no_region_found = (absolute_start == -1 or absolute_end == -1) 
        
        # --------------------------- 
        # Left Tile: Full View 
        # --------------------------- 
        ax1.plot(t_post_burn, amp_post_burn, color='black', linewidth=1.5, alpha=0.8, label="Mode Amplitude") 
        ax1.set_yscale('log') 
        ax1.set_title(rf"Full View: $\lambda$ = {dom_scale:.2e} m", fontsize=14, fontweight='bold') 
        ax1.set_xlabel("Time (s)", fontsize=12) 
        ax1.set_ylabel("Amplitude", fontsize=12) 
        ax1.grid(True, linestyle='--', alpha=0.4) 
        
        # --------------------------- 
        # Right Tile: Zoomed View 
        # --------------------------- 
        ax2.plot(t_post_burn, amp_post_burn, color='black', linewidth=1.5, marker='o', markersize=5, alpha=0.8, label="Mode Amplitude") 
        ax2.set_yscale('log') 
        ax2.set_title("Zoomed View: Linear Fit Region", fontsize=14, fontweight='bold') 
        ax2.set_xlabel("Time (s)", fontsize=12) 
        ax2.grid(True, linestyle='--', alpha=0.4) 
        
        # --------------------------- 
        # Apply Fits (if found) 
        # --------------------------- 
        if not no_region_found: 
            t_linear = self.times[absolute_start:absolute_end] 
            amp_linear = self.valid_a_x_t[absolute_start:absolute_end, amp_dom_idx] 
            if len(t_linear) > 1: 
                # Recalculate fit line for plotting 
                log_amp_linear = np.log(amp_linear + 1e-12) 
                slope, intercept = np.polyfit(t_linear, log_amp_linear, 1) 
                fit_line = np.exp(slope * t_linear + intercept) 
                
                # Add fit to Left Plot 
                ax1.axvspan(self.times[absolute_start], self.times[max(absolute_end - 1, absolute_start)], color='#2ca02c', alpha=0.2, label="Fit Region") 
                ax1.plot(t_linear, fit_line, color='red', linestyle='--', linewidth=2.5, label=rf"Linear Fit ($\gamma$={slope:.2e})") 
                
                # Add fit to Right Plot 
                ax2.axvspan(self.times[absolute_start], self.times[max(absolute_end - 1, absolute_start)], color='#2ca02c', alpha=0.2) 
                ax2.plot(t_linear, fit_line, color='red', linestyle='--', linewidth=2.5, label="Linear Fit") 
                
                # Dynamically set bounds for the Zoomed Plot (adding a buffer of 5 data points on each side) 
                left_buffer = 20 
                right_buffer = 5 
                zoom_start_idx = max(self.burn_in, absolute_start - left_buffer) 
                zoom_end_idx = min(len(self.times) - 1, absolute_end + right_buffer) 
                ax2.set_xlim(self.times[zoom_start_idx], self.times[zoom_end_idx]) 
                
                # Calculate Y limits dynamically based on the zoomed region to prevent squishing 
                y_zoom_data = self.valid_a_x_t[zoom_start_idx:zoom_end_idx+1, amp_dom_idx] 
                if len(y_zoom_data) > 0: 
                    y_min, y_max = np.min(y_zoom_data), np.max(y_zoom_data) 
                    log_min, log_max = np.log10(y_min + 1e-12), np.log10(y_max + 1e-12) 
                    margin = max((log_max - log_min) * 0.15, 0.1) # Add 15% vertical margin 
                    ax2.set_ylim(10**(log_min - margin), 10**(log_max + margin)) 
        else: 
            ax1.set_title(rf"Full View: $\lambda$ = {dom_scale:.2e} m (No Fit Found)", fontsize=14, fontweight='bold') 
            ax2.set_title("Zoomed View (No Fit Found)", fontsize=14, fontweight='bold') 
            
        # Clean up styling for both subplots 
        for ax in [ax1, ax2]: 
            ax.spines['top'].set_visible(False) 
            ax.spines['right'].set_visible(False) 
            ax.legend(loc='lower right' if ax == ax1 else 'best') 
            
        plt.suptitle(f"Simulation {self.simulation}: Dominant Mode & Linear Fit", fontsize=16, fontweight='bold', y=1.02) 
        plt.tight_layout() 
        if save: 
            os.makedirs(save_dir, exist_ok=True) 
            plt.savefig(os.path.join(save_dir, f"{self.simulation}_dominant_mode_fit.png"), dpi=300, bbox_inches='tight') 
        if show: 
            plt.show() 
        else: 
            plt.close()
