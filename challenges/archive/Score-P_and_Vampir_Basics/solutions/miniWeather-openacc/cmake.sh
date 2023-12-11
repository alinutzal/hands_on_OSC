#!/bin/bash

source ${MODULESHOME}/init/bash
module purge
module load DefApps nvhpc/21.11 cuda parallel-netcdf cmake
module unload darshan-runtime
module load scorep otf2 cubew 

export TEST_MPI_COMMAND="jsrun -n 1 -c 1 -a 1 -g 1"

source cmake_clean.sh

SCOREP_WRAPPER=off cmake -DCMAKE_CXX_COMPILER=scorep-mpicxx                                                                     \
      -DCXXFLAGS="-O3 -Mvect -DNO_INFORM -std=c++11 -I${OLCF_PARALLEL_NETCDF_ROOT}/include"                \
      -DLDFLAGS="-L${OLCF_PARALLEL_NETCDF_ROOT}/lib -lpnetcdf"                                         \
      -DOPENACC_FLAGS:STRING="-acc -gpu=cc70,fastmath,loadcache:L1,ptxinfo -Minfo=accel"               \
      -DNX=200                                                                                         \
      -DNZ=100                                                                                         \
      -DSIM_TIME=1000                                                                                  \
      -DOUT_FREQ=2000                                                                                  \
      .
