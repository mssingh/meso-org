# Codes related to the production of datasets used in this project.

## ERA5

Code used to generate ERA5 daily means over 2 and 5 degree mesoscale grids.
- CDO scripts to downsample full resolution single and pressure level ERA5 from rt52 to 1 degree daily means (already complete before this project)
- Python scripts to further downsample to 2 and 5 degree resolutions. A separate script is included for saturation fraction as it requires both 3D T and q data.
  