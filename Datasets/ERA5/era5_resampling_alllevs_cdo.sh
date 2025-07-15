#!/bin/bash
#PBS -P w40
#PBS -q normalbw
#PBS -l walltime=03:00:00
#PBS -l ncpus=2
#PBS -l mem=9GB
#PBS -l jobfs=20GB
#PBS -l storage=gdata/hh5+gdata/k10+gdata/rt52+scratch/k10

# for operating on data on all levels, e.g. pressure or theta level data

module load cdo/2.0.5

# change this to the ERA5 variable you want to use
VAR=z

FILES=/g/data/rt52/era5/pressure-levels/reanalysis/$VAR
OUTDIR=/g/data/k10/cr7888/era5_daily_means/${VAR}all/

# grid to interpolate to
GRID=/home/565/cr7888/PhD/phd_analysis/climatology/cdo/1_deg_grid_75.txt

for year in {1979..2021};  # 1979..2021
do
    for month in {01,02,03,04,05,06,07,08,09,10,11,12};
    do
        # select 60S - 60N, regrid to 1x1 deg, take daily mean (in that order)  r360x180 
        cdo -L -daymean -remapcon,$GRID -sellonlatbox,-180,180,-75,75 -sellevel,100/1000 $FILES/$year/${VAR}_era5_oper_pl_$year$month\01*.nc /scratch/k10/cr7888/${VAR}all/$year$month.nc
    done
    # concat monthly files into yearly
    cdo -b f32 mergetime /scratch/k10/cr7888/${VAR}all/$year*.nc $OUTDIR/era5_${VAR}_daily_mean_${year}.nc
done
