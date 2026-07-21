#!/bin/bash

gfortran -O2 -std=legacy -fallow-argument-mismatch -ffree-line-length-none \
  analytical_solution_viscoelastic_2D_plane_strain_Carcione_correct_with_1_over_L.f90 \
  -o analytical_solution

./analytical_solution 500 500
./analytical_solution 0 500
./analytical_solution 500 0
./analytical_solution '-500' 500
