function outputStates = propagate( inputBias, weights )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    outputNetInputs = weights * inputBias;
    outputStates = sigmoidFunc(outputNetInputs);
end

