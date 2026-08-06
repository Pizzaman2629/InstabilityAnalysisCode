""" 
ablation_analysis.py Code created by Lavya. 
Uses Chimera Reader (added to repository) 
Primary analysis method, more trustworth than simultion_analysis.py.

Uses ablation front tracking to calculate growth rates, etc.
""" 
import numpy as np 
import reader 
import os 
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm 
from scipy.integrate import simpson 
from scipy.fft import fft, fftfreq 
from scipy.signal import savgol_filter, find_peaks 
from scipy.interpolate import interp1d 

class Front_Sim(): 
    def __init__(self, simulation, ROOTDIR, project, start_step, final_step, step_interval, dump_freq, burn_in=0, min_window=10, r2_threshold = 0.98, axial="x", transverse="y", depth="z", solid_drop_threshold=0.5, breakout=False, breakout_threshold=0.03, cell_threshold=0, boundary_pad_cells=3, edge_buffer_cells=3, terminate_on_breakout=True, min_x_search=None, vis_front=True, vis_front_step=200, search_right_cells=10, target_thickness=100e-6, streak=False, streak_domain=None, streak_dump=None, streakbreakout=False, streakbreakout_domain=None, time_file=False, time_stop=None, window_sweep_points=5, window_sweep_stride=1): 
        """
        Class: Front_Sim. 
        Used to track the ablation front and do the relevant data processing. 

        Includes:
        1.) Growth Rate Calculations based on linear region detection. 
        2.) Sensitivity studies for fit certainties.
        """ 
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
            self.streak_domain = streak_domain + simulation + "Ablation_Front" 
            self.streak_dump = streak_dump if streak_dump is not None else dump_freq 
            
        #Fallbacks 
        if self.streak and self.streak_domain is None: 
            raise ValueError("streak=True requires streak_domain (Streak file name) to be set, e.g. 'streaks/xy0003Ablation_Front'.") 
            
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
            
        if time_stop is not None:
            valid_t_idx = np.where(self.times <= time_stop)[0]
            self.times = self.times[valid_t_idx]
            self.timesteps = self.timesteps[valid_t_idx]
            
        #Create timesteps to visualize the 2D density. 
        self.vis_front = vis_front 
        self.vis_front_timesteps = np.arange(start_step, final_step, vis_front_step) 
        
        #Assign helpful variables. 
        self.burn_in = burn_in 
        self.min_window = min_window 
        self.r2_threshold = r2_threshold 
        self.solid_drop_threshold = solid_drop_threshold 
        self.boundary_pad_cells = boundary_pad_cells 
        self.edge_buffer_cells = edge_buffer_cells 
        self.terminate_on_breakout = terminate_on_breakout 
        self.min_x_search = min_x_search 
        self.search_right_cells = search_right_cells 
        self.target_thickness = target_thickness 

        #Window sensitivity study settings (end-index sweep for the dominant mode fit).
        self.window_sweep_points = window_sweep_points
        self.window_sweep_stride = window_sweep_stride 
        
        #Assign variables for integration, interface and slicing. 
        self.interface = transverse 
        self.slice = depth 
        self.integration = axial 
        self.axial = axial 
        self.transverse = transverse 
        self.depth = depth 
        
        #Assign breakout related variables. 
        self.breakout = breakout 
        self.breakout_threshold = breakout_threshold 
        self.cell_threshold = cell_threshold 
        self.breakout_densities = None 
        self.breakout_pct_drop = None 
        self.breakout_time = None 
        self.dx_uniform = None 
        self.solid_boundary_idx = None 
        self.solid_boundary_x = None 
        self.breakout_index = None 
        
        #Load in axially integrated rho as a function of time, the interface grid and the domain size. 
        self.front_t, self.xc, self.yc = self.data_loader() 
        
        self.wavelength = (self.yc.max() - self.yc.min())/2 
        self.amp_t = np.abs(np.max(self.front_t, axis=1) - np.min(self.front_t, axis=1)) 
        
        #IMPORTANT: Raw Data Processing.
        #Scan for massive grid/vacuum drops on the RAW data before any filtering.
        #If it drops by more than 1 OOM and recovers later, bridge it smoothly.
        i = 0
        #Stepping using a while loop. (NOTE: This can maybe be optimized somehow for compute, but its fine for now)
        #The while loop ensures stepping only when necessary with the loop terminating as soon as a bridge is complete.
        while i < len(self.amp_t) - 1:
            val_curr = max(self.amp_t[i], 1e-16) #Current amplitude value
            val_next = max(self.amp_t[i+1], 1e-16) #Next amplitude value.

            #Check if there is a drop of more than a 1 order of magnitude.
            if np.log(val_curr) - np.log(val_next) > np.log(10):
                #Recovery index is where the value first gets out of the valley of despair.
                recovery_idx = -1
                for j in range(i + 1, len(self.amp_t)):
                    if max(self.amp_t[j], 1e-16) >= val_curr:
                        recovery_idx = j
                        break

                #If the recovery index is not the final index, bridge from the start index to the recovery index.
                if recovery_idx != -1:
                    log_start = np.log(val_curr)
                    log_end = np.log(max(self.amp_t[recovery_idx], 1e-16))
                    for k in range(i + 1, recovery_idx):
                        interp_log = log_start + (log_end - log_start) * ((k - i) / (recovery_idx - i))
                        self.amp_t[k] = np.exp(interp_log)
                    i = recovery_idx
                else:
                    i += 1
            else:
                i += 1

        #Loop through all indices in the raw amplitude and check if there are any flat areas. 
        #If there are flat areas, then substitute them with a straight line going to the next non-flat area.
        i = 0
        while i < len(self.amp_t) - 1:
            if self.amp_t[i] == self.amp_t[i+1]:
                plateau_end = i + 1
                #Walk forward to find the end of the flat plateau
                while plateau_end < len(self.amp_t) and self.amp_t[plateau_end] == self.amp_t[i]:
                    plateau_end += 1
                
                #If the plateau ends before the array does, we linearly interpolate the gap
                if plateau_end < len(self.amp_t):
                    val_start = self.amp_t[i]
                    val_end = self.amp_t[plateau_end]
                    for k in range(i + 1, plateau_end):
                        fraction = (k - i) / (plateau_end - i)
                        self.amp_t[k] = val_start + fraction * (val_end - val_start)
                i = plateau_end
            else:
                i += 1

        #Smooth out grid-snapping jitter of ~2*dx size to create a continuous curve.
        #Large jumps (like initialization cliffs) bypass the filter and stay raw.
        self.smoothed_amp_t = np.copy(self.amp_t)
        smooth_window = min(self.min_window, len(self.amp_t))
        if smooth_window % 2 == 0: smooth_window -= 1
        if smooth_window > 3:
            base_smoothed = savgol_filter(self.amp_t, window_length=smooth_window, polyorder=2)
            last_valid_amp = max(self.amp_t[0], 1e-16)  #Track the last safe amplitude
            for i in range(len(self.amp_t)):
                if np.abs(self.amp_t[i] - base_smoothed[i]) <= 2.5 * self.dx_uniform:
                    if base_smoothed[i] <= 1e-16:
                        #Prevent cavernous drops: hold the last valid value instead of dipping negative
                        self.smoothed_amp_t[i] = last_valid_amp
                    else:
                        self.smoothed_amp_t[i] = base_smoothed[i]
                        last_valid_amp = base_smoothed[i]
                else:
                    self.smoothed_amp_t[i] = self.amp_t[i]
                    if self.amp_t[i] > 1e-16:
                        last_valid_amp = self.amp_t[i]
                    
        self.valid_a_x_t = self.smoothed_amp_t.reshape(-1, 1) 
        self.raw_a_x_t = self.amp_t.reshape(-1, 1)
        self.wavelengths_m = np.array([self.wavelength]) 
        
        #Get growth rates using linear region detector 
        self.growth_rates, self.start_indices, self.end_indices = \
            self.compute_linear_regions(self.times, self.valid_a_x_t, self.burn_in, self.min_window) 
            
        #Amplitude dominant mode is the mode where the wavelength equals domain size (for the simulations we are concerned with, this can be changed later!) 
        self.amp_dom_idx = 0 
        
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

        #Window sensitivity study: only performed/saved for the amplitude dominant mode
        #Vary the end index and get a series of possible slopes for a known linear fit.
        #Take the standard deviation of the slope to get what we need.
        if self.start_indices[self.amp_dom_idx] != -1 and self.end_indices[self.amp_dom_idx] < len(self.times): 
            self.dom_a_growth_err = self.compute_window_sensitivity( 
                self.amp_dom_idx, 
                self.start_indices[self.amp_dom_idx], 
                self.end_indices[self.amp_dom_idx], 
                n_points=self.window_sweep_points, 
                stride=self.window_sweep_stride 
            ) 
        else: 
            self.dom_a_growth_err = np.nan 
            
        #Get the data for the max growth mode, if data doesn't exist, set to NaN value. 
        growth_dom_idx = 0 
        self.dom_g = self.valid_a_x_t[:, growth_dom_idx] 
        self.dom_g_growth = self.growth_rates[growth_dom_idx] 
        self.dom_g_lambda = self.wavelengths_m[growth_dom_idx] 
        if self.start_indices[growth_dom_idx] != -1 and self.end_indices[growth_dom_idx] < len(self.times): 
            self.dom_g_linstart = self.times[self.start_indices[growth_dom_idx]] 
            self.dom_g_linend = self.times[self.end_indices[growth_dom_idx]] 
        else: 
            self.dom_g_linstart = None 
            self.dom_g_linend = None 
            
        self.growth_rate = self.dom_a_growth 
        self.lin_start_idx = self.start_indices[self.amp_dom_idx] 
        self.lin_end_idx = self.end_indices[self.amp_dom_idx] 
        self.lin_start_t = self.dom_a_linstart 
        self.lin_end_t = self.dom_a_linend 

    #### CALCULATION: Finds the solid boundary and pops it out (for VTI Mode) ####
    def _find_solid_boundary(self, density_2d, coord, axis): 
        """
        Function to find the solid boundary near x = 0 and pop it out. 
        Only useful/valid in VTI Mode.
        """
        sum_axis = 1 - axis 
        profile_1d = np.sum(density_2d, axis=sum_axis) 
        log_profile = np.log(np.abs(profile_1d) + 1e-30) 
        valid_indices = np.where(coord >= 0.0)[0] 
        if len(valid_indices) == 0: 
            print(f"Simulation {self.simulation}: no positive coords found. Mask failed.") 
            return -1, coord.min() - 1.0 
        start_idx = valid_indices[0] 
        search_end_idx = min(start_idx + self.search_right_cells, len(coord)) 
        search_region = np.arange(start_idx, search_end_idx) 
        if search_region.size < 2: 
            return -1, coord.min() - 1.0 
        grad_1d = np.gradient(log_profile[search_region], coord[search_region]) 
        abs_grad = np.abs(grad_1d) 
        peak_local_idx = np.argmax(abs_grad) 
        peak_grad = abs_grad[peak_local_idx] 
        scan_limit = min(peak_local_idx + 20, len(search_region)) 
        drop_local_idx = peak_local_idx 
        for i in range(peak_local_idx, scan_limit): 
            if abs_grad[i] > 0.05 * peak_grad: 
                drop_local_idx = i 
        drop_idx = search_region[drop_local_idx] 
        print(f"LOG MASK | Simulation {self.simulation}: solid/vacuum boundary masked at coord = {coord[drop_idx]:.4e} (index {drop_idx}).") 
        return drop_idx, coord[drop_idx] 

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
        expected_len = surviving_coord.shape[0] 
        
        #Go 1 micron to the sides of x = 0 to get the dx value. 
        #NOTE: This works for a feathered grid which is uniform near x = 0, otherwise it does NOT work and should be changed! 
        idx0_init = np.argmin(np.abs(surviving_coord - (-1.0e-6))) 
        if idx0_init < len(surviving_coord) - 1: 
            self.dx_uniform = np.abs(surviving_coord[idx0_init + 1] - surviving_coord[idx0_init]) 
        else: 
            self.dx_uniform = np.abs(surviving_coord[idx0_init] - surviving_coord[idx0_init - 1]) 
            
        #Initialize empty lists for data loading. 
        front_t = [] 
        breakout_densities = [] 
        kept_timesteps = [] 
        kept_times = [] 
        exceed_count = 0 
        
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
                front_x = streak_df.iloc[idx, :].to_numpy(dtype=float) 
                #Fallbacks to ensure axial rho matches grid sizes. 
                if len(front_x) == expected_len + 1 and np.isnan(front_x[-1]): 
                    front_x = front_x[:-1] 
                elif len(front_x) == expected_len + 1 and not np.isnan(front_x[-1]): 
                    front_x = front_x[1:] 
                    
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
                    
                idx0 = np.argmin(np.abs(coord - (-1.0e-6))) 
                #The weird sign magic here keeps it negative. 
                interface_slice = np.take(density_2d, idx0, axis=axis) 
                breakout_density = np.mean(np.log(np.abs(interface_slice) + 1e-30)) 
                #Get mean value at that point and track it for reference. 
                breakout_densities.append(breakout_density) 
                
                if idx == 0: 
                    breakout_reference = breakout_density 
                    
                self.solid_boundary_idx, self.solid_boundary_x = self._find_solid_boundary(density_2d, coord, axis) 
                pct_drop = (breakout_reference - breakout_density) / breakout_reference 
                
                mask = (coord >= self.solid_boundary_x) & (coord <= self.solid_boundary_x + self.target_thickness) 
                mask_indices = np.where(mask)[0] 
                if mask_indices.size > self.boundary_pad_cells: 
                    mask[mask_indices[:self.boundary_pad_cells]] = False 
                    
                coord_masked = coord[mask] 
                grad_x = np.gradient(density_2d, coord, axis=axis) 
                
                if coord_masked.shape[0] < max(3, 2 * self.edge_buffer_cells + 1): 
                    raise ValueError(f"Simulation {self.simulation}: too few axial points remain in the target bounds at timestep {t}.") 
                    
                grad_mag = np.compress(mask, grad_x, axis=axis) 
                
                n_axial = grad_mag.shape[axis] 
                search_lo = self.edge_buffer_cells 
                search_hi = n_axial - self.edge_buffer_cells 
                if search_hi <= search_lo: 
                    search_lo, search_hi = 0, n_axial 
                    
                sl = [slice(None), slice(None)] 
                sl[axis] = slice(search_lo, search_hi) 
                
                front_idx_local = np.argmin(grad_mag[tuple(sl)], axis=axis) 
                front_idx = front_idx_local + search_lo 
                front_x = coord_masked[front_idx] 
                
                if self.vis_front and t in self.vis_front_timesteps: 
                    print(f"Visualizing 2D gradients and front for timestep: {t}") 
                    self._front_validation_plot(surviving_coord, front_x, t, grad_mag=grad_mag, xc_masked=coord_masked) 
                    
                if pct_drop > self.breakout_threshold: 
                    exceed_count += 1 
                else: 
                    exceed_count = 0 
                    
                if exceed_count > self.cell_threshold: 
                    self.breakout_time = self.times[idx] 
                    self.breakout_index = idx 
                    print(f"Simulation {self.simulation}: breakout at t = {self.breakout_time:.4e} s (pct drop {pct_drop:.4f})") 
                    if self.terminate_on_breakout: 
                        break 
                        
            #Fallbacks incase there is a mismatch in sizes. 
            if front_x.shape[0] != expected_len: 
                raise ValueError(f"Inconsistent grid at timestep {t}: front_x has length {front_x.shape[0]}, expected {expected_len}.") 
            #Set the NaN values which can arise from fortran to zero. 
            front_x = np.nan_to_num(front_x, nan=0.0) 
            #Append to the list. 
            front_t.append(front_x) 
            kept_timesteps.append(t) 
            kept_times.append(self.times[idx]) 
            
        self.timesteps = np.array(kept_timesteps) 
        self.times = np.array(kept_times) 
        
        #Make the breakout densities list into array for better manipulation. 
        breakout_densities = np.array(breakout_densities) 
        #Breakout logic (pretty straightforward tbh) 
        if len(breakout_densities) > 0 and self.streak: 
            breakout_reference = breakout_densities[0] 
            pct_drop = (breakout_reference - breakout_densities) / breakout_reference 
            above_thresh = np.where(pct_drop > self.breakout_threshold)[0] 
            self.breakout_densities = breakout_densities 
            self.breakout_pct_drop = pct_drop 
            if above_thresh.size > 0: 
                self.breakout_time = self.times[above_thresh[0]] 
            else: 
                self.breakout_time = None 
        elif not self.streak: 
            self.breakout_densities = breakout_densities 
        #If no breakout is detected, set to None 
        else: 
            self.breakout_densities = None 
            self.breakout_pct_drop = None 
            self.breakout_time = None 
            
        return np.array(front_t), coord, surviving_coord #Return all important values 

    #### PLOT: Validation of Ablation Front ####
    def _front_validation_plot(self, yc, front_x, t, grad_mag=None, xc_masked=None): 
        """ Plots the 2D log-gradient magnitude heatmap alongside the extracted front profile. """ 
        if grad_mag is not None and xc_masked is not None: 
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5)) 
            Y, X = np.meshgrid(yc, xc_masked) 
            c = ax1.pcolormesh(Y, X, grad_mag, shading='auto', cmap='viridis') 
            ax1.set_title(f"Log-Density Gradient Magnitude (t={t})", fontsize=12, fontweight='bold') 
            ax1.set_xlabel(f"Transverse ({self.transverse})") 
            ax1.set_ylabel(f"Axial ({self.axial})") 
            fig.colorbar(c, ax=ax1, label="|∇(log ρ)|") 
            ax2.plot(yc, front_x, color='#1f77b4', linewidth=2, marker='o', markersize=3) 
            ax2.grid(True, linestyle='--', alpha=0.6) 
            ax2.set_title(f"Extracted Ablation Front Position", fontsize=12, fontweight='bold') 
            ax2.set_xlabel(f"Transverse ({self.transverse})") 
            ax2.set_ylabel(f"Front Position ({self.axial})") 
            plt.tight_layout() 
            plt.show() 
        else: 
            plt.figure(figsize=(8, 5)) 
            plt.plot(yc, front_x, color='#1f77b4', linewidth=2, marker='o', markersize=3) 
            plt.grid(True, linestyle='--', alpha=0.6) 
            plt.title(f"Simulation {self.simulation}: Ablation Front Position (t={t})", fontsize=13, fontweight='bold') 
            plt.xlabel(f"Transverse ({self.transverse})", fontsize=12) 
            plt.ylabel(f"Front Position ({self.axial})", fontsize=12) 
            plt.tight_layout() 
            plt.show() 

    #### CALCULATION: Linear Region Detector #### 
    def find_anchored_linear_region(self, t_sliced, log_amp_sliced, raw_log_amp=None, min_window=10, r2_threshold=0.98): 
        """ 
        Function to calculate the linear region for a singular mode. 
        
        NOTE: This is probably THE most iffy function, so when doing analysis make sure to optimize it nicely.
         
        NOTE: Some of the methods here are geared towards the specific dataset we are working (for example the unsafe start procedure)
            These need to be given attention to.
        """ 
        #Get the number of tiem points. 
        n_points = len(t_sliced) 
        #If points are less than minimum window points, give a nan result (fallbacks) 
        if n_points <= min_window: 
            return 0, max(0, n_points - 1), np.nan 
            
        #Get the RAW amplitude data, not filtered. This will be used later to find huge cliffs without filter messing things up.
        check_amp = raw_log_amp if raw_log_amp is not None else log_amp_sliced

        #Setting the floor for grid clipping.
        #4e-07 was chosen as it is approximately 2*fallback dx which is needed to resolve ablation front.
        #Anything under this is most likely the front moving within a cell.
        grid_floor_log = np.log(4e-07 + 1e-16) 
        
        #Data can many times have initialization cliffs which are very bad. 
        #First step is to scan the data and start only when these cliffs are found. 
        #These cliffs generally have an increase of around 3 orders of magnitude within a single step. 
        cliff_end = 0
        step_window = min(3, len(check_amp) - 1) #Step 3 times atleast.
        
        if step_window > 0:
            for i in range(len(check_amp) - step_window):
                #Check for > 3 OOMs (np.log(1000) ~= 6.9) across the 3-step window
                if np.abs(check_amp[i+step_window] - check_amp[i]) > np.log(1000):
                    cliff_end = i + step_window
                #Also check single step for > 2 OOMs just in case
                elif np.abs(check_amp[i+1] - check_amp[i]) > np.log(100):
                    cliff_end = max(cliff_end, i + 1)
                    
        #Step 3 indices away from the last detected cliff to get a safe start position.
        if cliff_end > 0:
            safe_start = min(cliff_end, n_points - min_window - 1)
            
            #Switch back to the smoothed array for general flatness checks
            dy_smooth = np.gradient(log_amp_sliced)
            
            #Get maximum slope after trimming to the new start position. 
            if safe_start < len(dy_smooth): 
                max_slope_after = np.max(dy_smooth[safe_start:]) 
            else: 
                max_slope_after = 0 
                
            #Fallbacks (if there is no positive slope at all)
            if max_slope_after <= 0: 
                return 0, max(0, n_points - 1), np.nan 
                
            #Get a threshold to identify if the curve is flat or not. 
            flat_threshold = 0.05 * max_slope_after 
            
            #Step across indices to make sure we don't start at a flat point. 
            #Clamp to the first index where curve is not flat! 
            start_idx = safe_start 
            while start_idx < n_points - min_window: 
                if dy_smooth[start_idx] >= flat_threshold and check_amp[start_idx] > grid_floor_log + 1e-9: 
                    break 
                start_idx += 1 
                
            if start_idx >= n_points - min_window: 
                start_idx = max(0, n_points - min_window - 1) 
        else:
            #HARDCODE: Start directly at 0 if there is no unsafe start detected!
            #Step forward to bypass any initial negative slope (vacuum expansion)
            #AND skip flat sub-grid noise, but lock on immediately if it starts rising
            start_idx = 0
            while start_idx < n_points - min_window:
                is_subgrid_and_flat = (check_amp[start_idx] <= grid_floor_log + 1e-9) and (check_amp[start_idx + 1] <= check_amp[start_idx])
                is_decreasing = check_amp[start_idx + 1] < check_amp[start_idx]
                
                if is_subgrid_and_flat or is_decreasing:
                    start_idx += 1
                else:
                    break
            
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
            
            # SAFEGUARD: Prevent any accidental negative numbers from previous smoothing steps
            # AND clip out anything below grid resolution (~4e-07) to avoid fitting sub-grid noise
            amp_post_burn = np.maximum(amp_post_burn, 4e-07)
            
            #Lowered floor to 1e-16 to preserve true magnitudes of early simulation jumps!
            log_amp = np.log(amp_post_burn + 1e-16) 
            
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
            raw_log_amp_eval = log_amp_eval[:cap_idx + 1]
                
            #Get the linear region. 
            rel_start, rel_end, slope = self.find_anchored_linear_region( 
                t_eval, smoothed_log_amp, raw_log_amp=raw_log_amp_eval, min_window=min_window, r2_threshold=self.r2_threshold 
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

    #### CALCULATION: Window Sensitivity Study ####
    def compute_window_sensitivity(self, mode_idx, start_idx, end_idx, n_points=5, stride=1):
        """
        Function to compute window sensitivity.

        Takes a specific start index, end index pairing (a linear fit) and varies the end index. 
        Start index is not varied due to dataset specific reasons. 

        Slops for a bunch of end indices are collected and then their standard deviation is taken as the error bar for each fit.
        """
        amp_col = self.valid_a_x_t[:, mode_idx] #Get the amplitude for a given mode, only one exists for ablation tracking.
        amp_col = np.maximum(amp_col, 4e-07) #Get only the amplitudes above the fit.
        log_amp = np.log(amp_col + 1e-16) #Convert to log scale

        #Initialize the empty slope list
        slopes = [] 

        #Get the offsets
        offsets = range(-n_points * stride, n_points * stride + 1, stride) 

        #Loop over the offsets.
        for offset in offsets: 
            test_end = end_idx + offset 

            #Fallbacks: keep the window inside the data and long enough to fit. 
            if test_end <= start_idx + 1 or test_end >= len(self.times): 
                continue 
            #If the minimum window exceeds the one passed from outside the function, break.
            if test_end - start_idx < self.min_window:
                continue

            #Polyfit for the window to get the slope.
            t_window = self.times[start_idx:test_end] 
            y_window = log_amp[start_idx:test_end] 

            if len(t_window) < 2: 
                continue 

            slope, intercept = np.polyfit(t_window, y_window, 1) 

            #Only keep physically sensible (positive growth) slopes in the ensemble. 
            if slope <= 0: 
                continue 

            slopes.append(slope) 

        #Need at least 2 valid windows to say anything meaningful about the spread. 
        if len(slopes) < 2: 
            return np.nan 

        return np.std(slopes) #Return useful data :)

    #### PLOT: Plots raw amplitude and smooth amplitude with their linear fits. ####
    def amplitude_plot(self, save=False, show=True, save_dir="debug_plots"): 
        plt.figure(figsize=(10, 6)) 
        raw_amp = self.amp_t 
        smooth_amp = self.smoothed_amp_t
        valid = np.isfinite(smooth_amp) 
        plt.plot(self.times[valid], raw_amp[valid], color='gray', linewidth=1.0, alpha=0.4, label="Raw Amplitude (Grid Jitter)")
        plt.plot(self.times[valid], smooth_amp[valid], color='black', linewidth=1.5, alpha=0.8, marker='o', markersize=3, label="Smoothed Front Amplitude") 
        if self.lin_start_idx != -1 and self.lin_end_idx != -1 and self.lin_end_idx > self.lin_start_idx: 
            region_mask = valid[self.lin_start_idx:self.lin_end_idx] 
            t_lin = self.times[self.lin_start_idx:self.lin_end_idx][region_mask] 
            amp_lin = smooth_amp[self.lin_start_idx:self.lin_end_idx][region_mask] 
            if len(t_lin) > 1: 
                log_amp_lin = np.log(np.abs(amp_lin) + 1e-16) 
                slope, intercept = np.polyfit(t_lin, log_amp_lin, 1) 
                fit_line = np.exp(slope * t_lin + intercept) 
                plt.axvspan(self.times[self.lin_start_idx], self.times[max(self.lin_end_idx - 1, self.lin_start_idx)], color='#2ca02c', alpha=0.2) 
                plt.plot(t_lin, fit_line, color='red', linestyle='--', linewidth=2, label=rf"$\gamma$ = {self.growth_rate:.3e} s$^{{-1}}$") 
                plt.legend() 
        if self.breakout_time is not None: 
            plt.axvline(self.breakout_time, color='gray', linestyle=':', linewidth=2, label="Breakout (run terminated here)") 
            plt.legend() 
        plt.yscale('log') 
        plt.grid(True, linestyle='--', alpha=0.6, zorder=0) 
        plt.title(rf"Ablation Front Amplitude Evolution ($\lambda$ = {self.wavelength:.2e} m)", fontsize=14, fontweight='bold', pad=15) 
        plt.xlabel("Time (Seconds)", fontsize=12) 
        plt.ylabel("Front Perturbation Amplitude", fontsize=12) 
        ax = plt.gca() 
        ax.spines['top'].set_visible(False) 
        ax.spines['right'].set_visible(False) 
        plt.tight_layout() 
        if save: 
            os.makedirs(save_dir, exist_ok=True) 
            plt.savefig(os.path.join(save_dir, f"{self.simulation}_front_amp_plot.png"), dpi=300, bbox_inches='tight') 
        if show: 
            plt.show() 
        else: 
            plt.close() 

    #### PLOT: Plots a snapshot of the ablation front for a given index ####
    def front_snapshot_plot(self, timestep_index=-1, save=False, show=True, save_dir="debug_plots"): 
        front_x = self.front_t[timestep_index] 
        t = self.timesteps[timestep_index] 
        plt.figure(figsize=(8, 5)) 
        plt.plot(self.yc, front_x, 'o-', color='black', markersize=4, linewidth=1.5, label="Tracked front") 
        plt.grid(True, linestyle='--', alpha=0.6) 
        plt.legend() 
        plt.title(f"Simulation {self.simulation}: Front Position at timestep {t}", fontsize=13, fontweight='bold') 
        plt.xlabel(f"Transverse Coordinate ({self.transverse})", fontsize=12) 
        plt.ylabel(f"Front Position ({self.axial})", fontsize=12) 
        plt.tight_layout() 
        if save: 
            os.makedirs(save_dir, exist_ok=True) 
            plt.savefig(os.path.join(save_dir, f"{self.simulation}_front_snapshot_t{t}.png"), dpi=300, bbox_inches='tight') 
        if show: 
            plt.show() 
        else: 
            plt.close() 

    #### PLOT: Plots a 3D ablation surface (x,y,t) ####
    def plot_3d_ablation_surface(self, save=False, show=True, save_dir="debug_plots"): 
        fig = plt.figure(figsize=(12, 8)) 
        ax = fig.add_subplot(111, projection='3d') 
        Y, T = np.meshgrid(self.yc, self.times) 
        surf = ax.plot_surface(Y, T, self.front_t, cmap='inferno', linewidth=0.5, edgecolors='k', alpha=0.9) 
        ax.set_title(f"3D Ablation Front Evolution ({self.simulation})", fontsize=14, fontweight='bold', pad=20) 
        ax.set_xlabel(f"Transverse Coordinate ({self.transverse})", fontsize=12, labelpad=10) 
        ax.set_ylabel("Time (s)", fontsize=12, labelpad=10) 
        ax.set_zlabel(f"Front Position ({self.axial})", fontsize=12, labelpad=10) 
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, label=f"Front Depth ({self.axial})", pad=0.1) 
        ax.view_init(elev=30, azim=225) 
        plt.tight_layout() 
        if save: 
            os.makedirs(save_dir, exist_ok=True) 
            plt.savefig(os.path.join(save_dir, f"{self.simulation}_3D_surface.png"), dpi=300, bbox_inches='tight') 
        if show: 
            plt.show() 
        else: 
            plt.close() 

    #### PLOT: Plot the Dominant Mode and Its Linear Fit #### 
    def dominant_mode_fit_plotter(self, save=False, show=True, save_dir="debug_plots"): 
        """ 
        Plots the dominant amplitude mode and its linear fit. 
        Left tile: Full time evolution. 
        Right tile: Zoomed in view of the linear fit region with data point markers. 
        """ 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6)) 
        amp_dom_idx = self.amp_dom_idx 
        dom_scale = self.wavelengths_m[amp_dom_idx] 
        # Extract data post burn-in 
        t_post_burn = self.times[self.burn_in:] 
        
        # Raw for background, smoothed for primary line
        raw_amp_post_burn = self.raw_a_x_t[self.burn_in:, amp_dom_idx]
        amp_post_burn = self.valid_a_x_t[self.burn_in:, amp_dom_idx] 
        
        absolute_start = self.start_indices[amp_dom_idx] 
        absolute_end = self.end_indices[amp_dom_idx] 
        no_region_found = (absolute_start == -1 or absolute_end == -1) 
        
        # --------------------------- 
        # Left Tile: Full View 
        # --------------------------- 
        ax1.plot(t_post_burn, raw_amp_post_burn, color='gray', linewidth=1.0, alpha=0.4, label="Raw Amplitude (Grid Jitter)") 
        ax1.plot(t_post_burn, amp_post_burn, color='black', linewidth=1.5, alpha=0.8, label="Smoothed Mode Amplitude") 
        ax1.set_yscale('log') 
        ax1.set_title(rf"Full View: $\lambda$ = {dom_scale:.2e} m", fontsize=14, fontweight='bold') 
        ax1.set_xlabel("Time (s)", fontsize=12) 
        ax1.set_ylabel("Amplitude", fontsize=12) 
        ax1.grid(True, linestyle='--', alpha=0.4) 
        
        # --------------------------- 
        # Right Tile: Zoomed View 
        # --------------------------- 
        ax2.plot(t_post_burn, raw_amp_post_burn, color='gray', linewidth=1.0, alpha=0.4, label="Raw Amplitude") 
        ax2.plot(t_post_burn, amp_post_burn, color='black', linewidth=1.5, marker='o', markersize=5, alpha=0.8, label="Smoothed Amplitude") 
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
                log_amp_linear = np.log(amp_linear + 1e-16) 
                slope, intercept = np.polyfit(t_linear, log_amp_linear, 1) 
                fit_line = np.exp(slope * t_linear + intercept) 
                
                # Add fit to Left Plot 
                ax1.axvspan(self.times[absolute_start], self.times[max(absolute_end - 1, absolute_start)], color='#2ca02c', alpha=0.2, label="Fit Region") 
                fit_label = rf"Linear Fit ($\gamma$={slope:.2e}" 
                if hasattr(self, "dom_a_growth_err") and np.isfinite(self.dom_a_growth_err): 
                    fit_label += rf" $\pm$ {self.dom_a_growth_err:.1e}" 
                fit_label += ")" 
                ax1.plot(t_linear, fit_line, color='red', linestyle='--', linewidth=2.5, label=fit_label) 
                
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
                    log_min, log_max = np.log10(y_min + 1e-16), np.log10(y_max + 1e-16) 
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

    # ===================================================================== 
    # DEBUG PLOTTING STUBS FOR BATCH PROCESSING COMPATIBILITY 
    # These prevent the brane_sweeper from crashing when it calls 
    # simulator methods that only make sense for Fourier analysis. 
    # ===================================================================== 

    def mode_plotter(self, save=False, show=True, save_dir="debug_plots", **kwargs): 
        """ Alias for amplitude_plot to maintain compatibility with brane_sweeper diagnostics """ 
        self.amplitude_plot(save=save, show=show, save_dir=save_dir) 

    def timing_line_map_plotter(self, save=False, show=True, save_dir="debug_plots", **kwargs): 
        """ Stub function: Not applicable for single spatial front tracking. """ 
        pass 

    def grid_diagonistics(self, save=False, show=True, save_dir="debug_plots", **kwargs): 
        """ Stub function: Not applicable for single spatial front tracking. """ 
        pass 

    def growth_rate_plotter(self, save=False, show=True, save_dir="debug_plots", **kwargs): 
        """ Stub function: Not applicable for single spatial front tracking. """ 
        pass
