% Main algorithm for MNIST classification

% Files for MNIST
% if you don't have these you can download them at:
% http://yann.lecun.com/exdb/mnist/ 
% unzip them using "gunzip -d *-ubyte.gz"
TESTIMG_FILE = "t10k-images-idx3-ubyte";
TESTLBL_FILE = "t10k-labels-idx1-ubyte";
TRAINIMG_FILE = "train-images-idx3-ubyte";
TRAINLBL_FILE = "train-labels-idx1-ubyte";

TRNN = 6000;
TSTN = 1000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Extract the images from MNIST files    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if exist('mnistTrainImg')
    [~,~,trainsize] = size(mnistTrainImg);
    [~,~,testsize]  = size(mnistTrainImg);
    if (trainsize ~= TRNN)||(testsize ~= TSTN)
        [mnistTrainImg, mnistTrainLbl] = mnistParse(TRAINIMG_FILE, ...
                                                TRAINLBL_FILE, TRNN, 0);
        [mnistTestImg, mnistTestLbl] = mnistParse(TESTIMG_FILE, ...
                                                TESTLBL_FILE, TSTN, 0);
    end
else
    [mnistTrainImg, mnistTrainLbl] = mnistParse(TRAINIMG_FILE, ...
                                                TRAINLBL_FILE, TRNN, 0);
    [mnistTestImg, mnistTestLbl] = mnistParse(TESTIMG_FILE, ...
                                                TESTLBL_FILE, TSTN, 0);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Train the letter classifier on MNIST data %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train = 0;

model = 'knn';

if train == 1
% neural network loss: 0.3425 (patternnet, 10 nodes, 10-fold crossval)
% K-Nearest Neighbors (4 NN) loss: training: 0.0841, testing: 0.03811
    [mnistMdl, mnistLoss]  = mnistTrain(mnistTrainImg, mnistTrainLbl, ...
                                            mnistTestImg, mnistTestLbl, ...
                                            model);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Find and detect letters in a sample image %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mser = 1;

if mser==1
    % image to classify
    image = imread("test_images/test4.png");
    I = rgb2gray(image);
    
    letters = textDetection(I);

    [~,~,nl] = size(letters);
    %imshow(letters(:,:,2));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Classify the detected letters %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
for i = 1:nl
    ltr = letters(:,:,i);
    
    subplot(3,nl,i)
    [pClass, pLoss] = mnistClassify( mnistMdl, ltr, model);
    plotNumber ( mnistTrainImg, mnistTrainLbl, pClass);
    title("Predicted Digit");

    subplot(3,nl,nl+i)
    imshow(letters(:,:,i));
    title("Actual Digit");

    subplot(3,nl,2*nl+i)
    bar(0:9,pLoss)
    title("Digit Probability")
end

%subplot(2,2,4)
%plot(1:10, mnistLoss(1,:), 1:10, mnistLoss(2,:));
%legend("Testing Loss", "Training Loss");
%xlabel("# of Nearest Neighbors")
%ylabel("Loss")
%title("Performance estimate for KNN Parameter Optimization")

%plotNumber( mnistTrainImg, mnistTrainLbl, [1 2 3 4 5 6]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Supplimentary Functions %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plotDigit(arr, ltrIdx)
% plotDigit - plots a digit of the MNIST training set
    % plot the 3d array
    imshow(flipud(arr(:,:,ltrIdx)));    
end

function [digit] = findDigit(dataset, labelset, label)
    idx = find(labelset == label);
    idx = idx(1); % just use the first one
    digit = dataset(:,:,idx);
end

function plotNumber(dataset, labelset, lblArr)
    [~,idx] = size(lblArr);
    word = [];
    for n = lblArr
        word = [word findDigit(dataset,labelset,n)];
    end
    imshow(word);
end
