function [deltas] = hiddenDeltas(outputDeltas,hiddenOutputs,outputWeights)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

sigmoidDeriv = hiddenOutputs .* (ones(size(hiddenOutputs)) - hiddenOutputs);
deltas = (outputWeights' * outputDeltas) .* sigmoidDeriv;

end

