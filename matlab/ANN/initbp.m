digitPats;
nPats = size(patterns,2);
nTrainingPats = 20;
nTestPats = 20;
nInputs = size(patterns,1);
nHidden = 10;
nOutputs = 4;

hiddenWeights = 0.5 * (rand(nHidden,nInputs+1) - ones(nHidden,nInputs+1) * 0.5);
outputWeights = 0.5 * (rand(nOutputs,nHidden+1) - ones(nOutputs,nHidden+1) * 0.5);

input = patterns;
target = zeros(nOutputs,nPats);
class = 1;
for pat = 1:nPats
    target(class,pat) = 1;
    class = class + 1;
    if class > nOutputs
        class = 1;
    end
end

eta = 0.1;

NEpochs = 1000;
totalNEpochs = 0;

minEpochsPerErrorPlot = 200;
errorsPerEpoch = zeros(1,minEpochsPerErrorPlot);
TestErrorsPerEpoch = zeros(1,minEpochsPerErrorPlot);
epochs = [1:minEpochsPerErrorPlot];