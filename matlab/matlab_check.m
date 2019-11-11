
clear; clc;

addpath('C:\Users\yurio\Documents\GitHub\orbital_matlabtools')

sv = [8.88099826e+04, -1.51900741e+08,  7.67777168e+06; 2.92911349e+01,  3.87094214e-03,  1.42259745e-01];

r = sv(1,1:3);
v = sv(2,1:3);

mu = 132712000000;

OE = OrbitalElements(r,v,mu)


