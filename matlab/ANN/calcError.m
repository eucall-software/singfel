function [sumSqrError,wrong] = calcError( targetStates, outputStates )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    error = outputStates - targetStates;        
    sumSqrError = dot(error,error);
    wrong = abs(round(outputStates)-targetStates);
end

%         targetStates = target(:,pat);
%         numWrong = numWrong + abs(round(outputStates)-targetStates);
%         error = outputStates - targetStates;
%         sumSqrError = sumSqrError + dot(error,error);