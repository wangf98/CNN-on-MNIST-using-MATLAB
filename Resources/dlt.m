%digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
 %   'nndatasets','DigitDataset');
addpath ../data/;
trainimages=LoadImage('../data/train-images-idx3-ubyte');
trainimages=reshape(trainimages,28,28,1,[]);
trainlabels=LoadLabel('../data/train-labels-idx1-ubyte');
trainlabels(trainlabels==0)=10;
testimages=LoadImage('../data/t10k-images-idx3-ubyte');
testimages=reshape(testimages,28,28,1,[]);
testlabels=LoadLabel('../data/t10k-labels-idx1-ubyte');
testlabels(testlabels==0)=10;

numTrainFiles = 1;
imdsTrain={trainimages;trainlabels};
imdsValidation={testimages;testlabels};
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(5,8)
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(5,32)
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',3, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress');

net = trainNetwork(trainimages,categorical(trainlabels),layers,options);
%print(net.Layers);
YPred = classify(net,testimages);
YValidation = categorical(testlabels);

accuracy = sum(YPred == YValidation)/numel(YValidation);