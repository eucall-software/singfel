function [deltas] = outputDeltas(output,target)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

sigmoidDeriv = output .* (ones(size(output)) - output);
deltas = (target - output) .* sigmoidDeriv;

end

