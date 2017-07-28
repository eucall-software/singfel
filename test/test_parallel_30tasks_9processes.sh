#!/bin/bash

mpirun -np 9 ../build/bin/radiationDamageMPI --inputDir pmi --outputDir /tmp --beamFile s2e.beam --geomFile s2e.geom --configFile /dev/null --pmiStartID 1 --pmiEndID 3 --numSlices 100 --sliceInterval 10 --numDP 10