from simulation_analysis import Single_Sim
from batch_processing import line_sweeper

#Setting up some parameters.
simulations = ["xy32", "xy33", "xy34"]
ROOTDIR = r"D:/Projects/ICF Summer Internship"
project = r"Paraview Data"
parameter_name = "Add Name Here"
parameters = [1, 2, 3]   


simulator = Single_Sim(simulation="xy0003", ROOTDIR=ROOTDIR, project=project,
                       start_step=0, final_step=100, step_interval=100, dump_freq=1.0e-11,
                       slice_dir = "z", interface = "y", integration = "x", 
                       vis_rho=True, vis_rho_step=100, 
                       breakout = False, breakout_threshold=0.03, cell_threshold=10)

"""
Plotting that can be done with a singular simulation
"""
#simulator.mode_plotter()
#simulator.timing_line_map_plotter()
#simulator.grid_diagonistics()
#simulator.growth_rate_plotter()

"""
Stuff we can dow ith the line sweeper, easily extendable to brane sweepers. 
"""
"""
# Lambda function acting as the simulator factory
simulator = lambda x: Single_Sim(simulation=x, ROOTDIR=ROOTDIR, project=project,
                       start_step=0, final_step=600, step_interval=10, dump_freq=1.0e-11,
                       slice_dir = "z", interface = "y", integration = "x", 
                       breakout=True, breakout_threshold=0.03, cell_threshold=10)

# Initializing the 1D line sweeper
sweeper = line_sweeper(
    simulator=simulator, 
    simulations=simulations, 
    parameters=parameters, 
    parameter_name=parameter_name
)

# Generating the line sweep plots
sweeper.dominant_mode(show=True)
sweeper.growth_mode(show=True)
sweeper.breakout_time_plot(show=True)
"""