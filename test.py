"""
test.py
Reads batch job CSV files to extract parameters and run indices,
calculates dynamic stop times based on simulation parameters,
and feeds the batched data into the brane_sweeper.
"""
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
from ablation_analysis import Front_Sim  #Imported to allow easy swapping b/w simulators.
from batch_processing import brane_sweeper
from simulation_analysis import Single_Sim

#### DIAGNOSTIC: Plots Fits for a Single Intensity ####
def overlay_amplitude_fits(sweeper, save=False, show=True):
    """
    Function to take all simulations with a single intensity and plot the dominant mode and its fits. 

    Very useful for broad fit diagnostics and to verify if the simulation has done its thing correctly.
    """
    print("Generating individual overlaid amplitude fits grouped by Peak Intensity...")
    
    # Locate column indices for our parameters
    try:
        group_idx = sweeper.parameter_names.index("Peak Intensity")
        wave_name = "Wavelength (μm)" if "Wavelength (μm)" in sweeper.parameter_names else "Wavelength"
        wave_idx = sweeper.parameter_names.index(wave_name)
    except ValueError as e:
        print(f"Skipping overlay plot: missing expected parameter names. {e}")
        return

    unique_groups = np.unique(sweeper.parameters[:, group_idx])

    for group_val in unique_groups:
        plt.figure(figsize=(10, 6))

        # Find all simulations matching this Intensity and sort by Wavelength
        sim_indices = np.where(sweeper.parameters[:, group_idx] == group_val)[0]
        sim_indices = sorted(sim_indices, key=lambda idx: sweeper.parameters[idx, wave_idx])

        cmap = plt.cm.turbo
        colors = cmap(np.linspace(0.05, 0.95, len(sim_indices)))
        
        min_t_fit = np.inf
        max_t_fit = -np.inf

        for j, sim_idx in enumerate(sim_indices):
            amp_arr = sweeper.dom_a[sim_idx]
            wave_val = sweeper.parameters[sim_idx, wave_idx]
            c = colors[j]

            # Reconstruct time array if not natively saved
            if hasattr(sweeper, 'times') and len(sweeper.times) > sim_idx:
                t_arr = sweeper.times[sim_idx]
            else:
                t_arr = np.arange(len(amp_arr)) * 1e-12

            # Plot full raw amplitude in the background
            plt.plot(t_arr, amp_arr, color=c, alpha=0.3, linewidth=1.5)

            t_start = sweeper.dom_a_linstart[sim_idx]
            t_end = sweeper.dom_a_linend[sim_idx]

            # Check if fit is valid
            if t_start is not None and t_end is not None and np.isfinite(t_start) and np.isfinite(t_end):
                # Update global bounds for the zoom
                if t_start < min_t_fit: min_t_fit = t_start
                if t_end > max_t_fit: max_t_fit = t_end

                mask = (t_arr >= t_start) & (t_arr <= t_end)
                t_lin = t_arr[mask]
                amp_lin = amp_arr[mask]

                if len(t_lin) > 1:
                    # PULL DIRECTLY FROM SWEEPER INSTEAD OF RECALCULATING
                    slope = sweeper.dom_a_growth[sim_idx]
                    
                    # We still need the intercept to draw the line on the graph
                    log_amp = np.log(amp_lin + 1e-12)
                    intercept = np.mean(log_amp - slope * t_lin)
                    fit_line = np.exp(slope * t_lin + intercept)

                    # Plot the bold fit line
                    plt.plot(t_lin, fit_line, color=c, linestyle='--', linewidth=2.5, 
                             label=rf"$\lambda$ = {wave_val:.1f} $\mu$m ($\gamma$={slope:.1e})")
                else:
                    plt.plot([], [], color=c, linestyle='-', linewidth=1.5, alpha=0.5, 
                             label=rf"$\lambda$ = {wave_val:.1f} $\mu$m (No Fit)")

        plt.yscale('log')
        plt.grid(True, linestyle='--', alpha=0.6, zorder=0)

        if sweeper.parameter_scales[group_idx] == 'log':
            group_title = f"Amplitude Evolutions (Peak Intensity: {group_val:.1e})"
            safe_val = f"{group_val:.1e}".replace('+', '').replace('.', '_')
        else:
            group_title = f"Amplitude Evolutions (Peak Intensity: {group_val})"
            safe_val = f"{group_val}".replace('.', '_')
            
        plt.title(group_title, fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Amplitude", fontsize=12)

        # --- Apply the Dynamic Zoom ---
        if min_t_fit < np.inf and max_t_fit > -np.inf:
            t_span = max_t_fit - min_t_fit
            if t_span == 0: t_span = 1e-10  # Fallback if only 1 data point found
            
            # Add a 20% visual margin around the linear region block
            margin = t_span * 0.2
            plt.xlim(max(0, min_t_fit - margin), max_t_fit + margin)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if save:
            plt.savefig(f"overlay_amplitude_intensity_{safe_val}.png", dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()

"""
Main class for all the data loading from CSV's and brane sweeping.
"""
def main():
    #Setting up directories for reader.
    ROOTDIR = r"/rds/general/project/cifs/live/ldg226/laser_speckle/"
    
    #List of projects incase there are simulations that need to be combined together.
    projects = [r"2d_scan1/", r"2d_scan2/"]

    #Parameter names. This can be extracted straight from CSV files (future extension maybe).
    parameter_names = ["Wavelength", "Contrast", "Peak Intensity", "Thickness"]
    parameter_scales = ["log", "linear", "log", "linear"]

    #Initializing global lists to hold the data needed for the brane sweeper.
    simulations_list = []
    parameters_list = []
    final_steps_dict = {}  #This dictionary allows for different simulation stop times.
    min_window_dict = {}   #Holds the dynamic minimum window assigned to each simulation.
    sim_lookup_dict = {}   #Maps unique names back to (actual_sim_name, project)

    #Loop over projects.
    for current_project in projects:
        #Find the CSV files (in a job directory)
        search_path = f"{ROOTDIR}{current_project}jobs/job_*.csv"
        csv_files = glob.glob(search_path)

        #CSV File detection fallback.
        if not csv_files:
            print(f"No CSV files found in '{current_project}'. Skipping...")
            continue

        #Console output.
        print(f"Found {len(csv_files)} job CSV files in {current_project}. Extracting runs...")

        csv_row_count = 0
        loaded_count = 0

        #Loop through all CSV files (this is for compatibility with multiple jobs)
        for csv_file in csv_files:
            #Load CSV File as dataframe. Each column is one parameter. First column is run index.
            df = pd.read_csv(csv_file)

            #Loop over different rows (to add to lists)
            for _, row in df.iterrows():
                csv_row_count += 1
                
                #Extract raw values (kept as float to preserve the .0 for the directory names!)
                run_index = float(row.iloc[0])
                wavelength = float(row.iloc[1])
                contrast = float(row.iloc[2])
                intensity = float(row.iloc[3])
                thickness = float(row.iloc[4])
                
                """
                EXAMPLE: If you want to isolate some specific runs.
                if wavelength <= 100e-6:
                    continue
                """

                #Construct original simulation name
                sim_name = f"xy{run_index}"
                dat_path = f"{ROOTDIR}{current_project}{sim_name}/{sim_name}.dat"

                #DIAGNOSTIC CHECK: Print exactly what fails so you know why it's missing
                if not os.path.exists(dat_path):
                    print(f"  [Warning] Missing or inaccessible file: {dat_path}")
                    continue

                loaded_count += 1

                #Create a unique sim name to avoid conflicts between projects with the same run index
                clean_project_name = current_project.replace("/", "").replace("\\", "")
                unique_sim_name = f"{clean_project_name}_{sim_name}"

                #Assign stop times (this is borrowed from the job script)
                if intensity < 1e19 or thickness > 25e-6:
                    time_stop = 3.0e-09
                else:
                    time_stop = 1.0e-09

                #Assign dynamic minimum window based on peak intensity thresholds
                #NOTE: This was optimized by trial and error and pain.
                if intensity < 1e19:
                    window_val = 15
                elif intensity < 1e20:
                    window_val = 10
                else:
                    window_val = 2

                #Dump normalization
                dump_norm = 1.0e-12
                final_step = int(time_stop / dump_norm) #Important to keep VTI and streak data consistent.

                #Append to global lists.
                simulations_list.append(unique_sim_name)
                parameters_list.append([wavelength, contrast, intensity, thickness])
                final_steps_dict[unique_sim_name] = final_step
                min_window_dict[unique_sim_name] = window_val
                sim_lookup_dict[unique_sim_name] = (sim_name, current_project)

        print(f"  -> Project {current_project}: Read {csv_row_count} rows from CSV, successfully found {loaded_count} valid .dat files.")

    if not simulations_list:
        print("No valid simulations found across any projects. Exiting.")
        return

    #Convert parameters list into array.
    parameters_matrix = np.array(parameters_list)

    # Console output
    print(f"\nSuccessfully loaded {len(simulations_list)} TOTAL simulations across all projects.")

    """
    Wrapper function for the simulator. This is used to allow for dynamic time stops.
    It decodes the unique_sim_name back into its original project and sim_name.
    """
    def simulator_factory(unique_sim_name):
        actual_sim_name, actual_project = sim_lookup_dict[unique_sim_name]
        return Single_Sim(
            simulation=actual_sim_name,
            ROOTDIR=ROOTDIR,
            project=actual_project,
            start_step=0,
            final_step=final_steps_dict[unique_sim_name],
            step_interval=100,  # VTI dump every 100ps
            dump_freq=1.0e-10,
            min_window=min_window_dict[unique_sim_name],
            slice_dir="z",          # Mapped from slice_dir="z"
            interface="y",      # Mapped from interface="y"
            integration="x",          # Mapped from integration="x"
            vis_rho=False,    # Mapped from vis_rho=False
            vis_rho_step=100, # Mapped from vis_rho_step=100
            breakout_threshold=0.03,
            cell_threshold=10,
            streak=True,
            streak_domain="streaks/",
            streak_dump=1e-12,
            time_file=True,
            burn_in = 0
        )

    #Running the brane sweeper.
    print("Initializing brane_sweeper...")
    sweeper = brane_sweeper(
        simulator=simulator_factory,
        simulations=simulations_list,
        parameters=parameters_matrix,
        parameter_names=parameter_names,
        debug=False, #NOTE: Debugging is very important for ensuring accuracy of analysis.
        debug_interval=2,
        debug_dir="debug_plots",
        parameter_scales=parameter_scales
    )

    #Output all the required plots.
    print("Generating sweeps...")
    overlay_amplitude_fits(sweeper, show=True, save=True)
    sweeper.tiled_summary_plots(show=True, save=True)
    
    print("Batch processing complete.")

if __name__ == "__main__":
    main()
