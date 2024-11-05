# Photometry_pipeline

Notebooks
- Framework_photom.ipynb
  - Highest level notebook
  - Framework photom loads in compressed cages for both manip and wheel, so you can ask questions intra and inter behavior.
  - Running linear regression per trial, generating barcharts across conditions (ie: behavior, early vs late, trial type)
  - The general structure is:
    - Parameters, Classes
    - Load compressed cage
    - Analysis and visualization
- Photometry_manipulandum_initial process.ipynb, Photometry_wheel_initial_process.ipynb 
  - These process raw data and generate “cages” as well as compressed versions
  - Framework_photom loads in these cages
  - Also have the functionality to load a pre-generated (pickled) cage
  - And to update (for instance the photometry cubes, random cubes, etc) and re-save (overwrite) those pickled folders 

Data Analysis Helperfuncs
- Behav_data_analysis
  - Helper functions for manipulandum behavior
- Dlc_helper
  - Helper function for deep lab cut analysis
- Preprocess_helper
  - Helper functions for 
- Metric_helper 
  - Stores different metrics to apply to trials
- Statistics_helper 
  - Has helpers for parsing (by mouse) and for stats (t stats)
  - Useful: “group_by_mouse_only” function 
- Stride_Stance_helper 
  - More detailed wheel behavior analysis with stride and stance
