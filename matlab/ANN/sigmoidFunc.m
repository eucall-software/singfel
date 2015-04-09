function output = sigmoidFunc(totalInput)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

output = 1 ./ [ones(size(totalInput))+exp(-totalInput)];

end

