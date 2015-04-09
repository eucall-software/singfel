ErrorsLastNEpochs = zeros(1,NEpochs);
TestErrorsLastNEpochs = zeros(1,NEpochs);
startEpoch = totalNEpochs + 1;

for epoch = 1:NEpochs
    sumSqrError = 0.0;
    sumSqrTestError = 0.0;
    outputWGrad = zeros(size(outputWeights));
    hiddenWGrad = zeros(size(hiddenWeights));
    for pat = 1:nTrainingPats
        % forward
        inp = [input(:,pat)',[1]]';
        hiddenNetInputs = hiddenWeights * inp;
        hiddenStates = sigmoidFunc(hiddenNetInputs);
        hidStatesBias = [[hiddenStates]',[1]]';
        outputNetInputs = outputWeights * hidStatesBias;
        outputStates = sigmoidFunc(outputNetInputs);
        
        % Backward
        targetStates = target(:,pat);
        error = outputStates - targetStates;
        sumSqrError = sumSqrError + dot(error,error);
        outputDel = outputDeltas(outputStates,targetStates);
        outputWGrad = outputWGrad + outputDel * hidStatesBias';
        hiddenDel = hiddenDeltas(outputDel,hidStatesBias,outputWeights);
        hiddenWGrad = hiddenWGrad + hiddenDel(1:nHidden,:) * inp';
    end
    outputWChange = eta * outputWGrad;
    outputWeights = outputWeights + outputWChange;
    hiddenWChange = eta * hiddenWGrad;
    hiddenWeights = hiddenWeights + hiddenWChange;
    
    for pat = (nTrainingPats+1):nPats
        inp = [input(:,pat)',[1]]';
        hiddenNetInputs = hiddenWeights * inp;
        hiddenStates = sigmoidFunc(hiddenNetInputs);
        hidStatesBias = [[hiddenStates]',[1]]';
        outputNetInputs = outputWeights * hidStatesBias;
        outputStates = sigmoidFunc(outputNetInputs);
        targetStates = target(:,pat);
        error = outputStates - targetStates;
        sumSqrTestError = sumSqrTestError + dot(error,error);
    end
    
    gradSize = norm([hiddenWGrad(:);outputWGrad(:)]);
    totalNEpochs = totalNEpochs + 1;
    MSE = sumSqrError/nTrainingPats;
    TestMSE = sumSqrTestError/nTestPats;
    if totalNEpochs == 1
        startError = MSE;
    end
    ErrorsLastNEpochs(1,epoch) = MSE;
    TestErrorsLastNEpochs(1,epoch) = TestMSE;
    fprintf(1,'%d MSError=%f, MSTestError=%f, |G|=%f\n',...
        totalNEpochs,MSE,TestMSE,gradSize);
end

clf
figure
if totalNEpochs > minEpochsPerErrorPlot
    epochs = [1:totalNEpochs];
end

errorsPerEpoch(1,startEpoch:totalNEpochs) = ErrorsLastNEpochs;
subplot(2,1,1)
axis([1 max(minEpochsPerErrorPlot,totalNEpochs) 0 startError]),
hold on
plot(epochs(1,1:totalNEpochs),errorsPerEpoch(1,1:totalNEpochs)),
title('Mean Squared Error on the Training Set'),
xlabel('Learning Epoch')
ylabel('MSE')

TestErrorsPerEpoch(1,startEpoch:totalNEpochs) = TestErrorsLastNEpochs;
subplot(2,1,2)
axis([1 max(minEpochsPerErrorPlot,totalNEpochs) 0 startError]),
hold on
plot(epochs(1,1:totalNEpochs),TestErrorsPerEpoch(1,1:totalNEpochs)),
title('Mean Squared Error on the Test Set'),
xlabel('Learning Epoch')
ylabel('MSE')
