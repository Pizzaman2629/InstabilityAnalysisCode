"""
data_loader_runner.py

Reads batch job CSV files to extract parameters and run indices, calculates 
dynamic stop times based on simulation parameters, and feeds the batched 
data into the brane_sweeper.
"""

import glob
import numpy as np
import pandas as pd
from simulation_analysis import Single_Sim
from batch_processing import brane_sweeper

"""
Main class for all the data loading from CSV's and brane sweeping.
"""
def main():
    #Setting up directories for reader.
    ROOTDIR = r"D:/Projects/ICF Summer Internship"
    project = r"Paraview Data"
    
    #Parameter names. This can be extracted straight from CSV files (future extension maybe).
    parameter_names = ["Wavelength", "Contrast", "Peak Intensity", "Thickness"]
    
    #Initializing global lists to hold the data needed for the brane sweeper.
    simulations_list = []
    parameters_list = []
    final_steps_dict = {}  #This dictionary allows for different simulation stop times.

    #Find the CSV files (in a job directory)
    csv_files = glob.glob("jobs/job_*.csv")

    #CSV File detection fallback.
    if not csv_files:
        print("No CSV files found matching 'jobs/job_*.csv'. Please check the path.")
        return

    #Console output.
    print(f"Found {len(csv_files)} job CSV files. Extracting runs...")

    #Loop through all CSV files (this is for compatibility with multiple jobs)
    for csv_file in csv_files:
        #Load CSV File as dataframe. Each column is one parameter. First column is run index,
        df = pd.read_csv(csv_file)
        
        #Loop over different rows (to add to lists)
        for _, row in df.iterrows():
            #Extract raw values
            run_index = row.iloc[0]
            wavelength = row.iloc[1]
            contrast = row.iloc[2]
            intensity = row.iloc[3]
            thickness = row.iloc[4]

            #Construct simulation name, this needs to be precise to work well with HPC naming convention.
            sim_name = f"xy{run_index}"
            
            #Assign stop times (this is borrowed from the job script)
            if intensity < 1e19 or thickness > 25e-6:
                time_stop = 3.0e-09
            else:
                time_stop = 1.0e-09

            #Dump normalization
            dump_norm = 1.0e-12 
            final_step = int(time_stop / dump_norm) #This is important to keep VTI and streak data consistent. 

            #Append to gloabl lists.
            simulations_list.append(sim_name)
            parameters_list.append([wavelength, contrast, intensity, thickness])
            final_steps_dict[sim_name] = final_step

    #Convert lists into arrays for the brane sweeper.
    parameters_matrix = np.array(parameters_list)

    #Console output
    print(f"Successfully loaded {len(simulations_list)} total simulations.")

    """
    Wrapper function for the simulator. This is used to allow for dynamic time stops.

    NOTE: Not yet configured for streaks, although it would be almost identical just changing a few flags around.
    """
    def simulator_factory(sim_name):
        return Single_Sim(
            simulation=sim_name, 
            ROOTDIR=ROOTDIR, 
            project=project,
            start_step=0, 
            final_step=final_steps_dict[sim_name], 
            step_interval=100,  #VTI dump every 100ps
            dump_freq=1.0e-12,  
            slice_dir="z", 
            interface="y", 
            integration="x", 
            vis_rho=True, 
            vis_rho_step=100, 
            breakout=True, 
            breakout_threshold=0.03, 
            cell_threshold=10
        )

    #Running the brane sweeper.
    print("Initializing brane_sweeper...")
    sweeper = brane_sweeper(
        simulator=simulator_factory,
        simulations=simulations_list,
        parameters=parameters_matrix,
        parameter_names=parameter_names,
        debug=True,
        debug_interval=100,
        debug_dir="debug_plots"
    )

    #Generating and outputting all required plots.
    print("Generating sweeps...")
    sweeper.dominant_mode(show=True, save=True)
    sweeper.growth_mode(show=True, save=True)
    sweeper.breakout_time_plot(show=True, save=True)
    print("Batch processing complete.")

if __name__ == "__main__":
    main()
