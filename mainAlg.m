% image to classify
image = "testImage.png";

% Files for MNIST
TESTIMG_FILE = "t10k-images-idx3-ubyte";
TESTLBL_FILE = "t10k-labels-idx1-ubyte";
TRAINIMG_FILE = "train-images-idx3-ubyte";
TRAINLBL_FILE = "train-labels-idx1-ubyte";

[mnistTrainImg, mnistTrainLbl] = mnistParse(TRAINIMG_FILE, TRAINLBL_FILE, 60000, 0);
[mnistTestImg, mnistTestLbl] = mnistParse(TESTIMG_FILE, TESTLBL_FILE, 10000, 0);

[mnistNet, mnistLoss]  = mnistTrain(mnistTrainImg, mnistTrainLbl, mnistTestImg, mnistTestLbl);

class = 3;

% find a digit from the test set
img = findDigit( mnistTestImg, mnistTestLbl, class);

figure
subplot(2,1,1)
pClass = mnistClassify( mnistNet, img);
plotNumber ( mnistTrainImg, mnistTrainLbl, pClass);
title("Predicted Digit");

subplot(2,1,2)
plotNumber ( mnistTrainImg, mnistTrainLbl, class);
title("Actual Digit");

%plotNumber( mnistTrainImg, mnistTrainLbl, [1 2 3 4 5 6]);

function plotDigit(arr, ltrIdx)
% plotDigit - plots a digit of the MNIST training set
    % plot the 3d array
    surf(flipud(arr(:,:,ltrIdx)), 'EdgeColor','interp');
    % look at the overview so it looks like a 2d image
    view(2)
end

function [digit] = findDigit(dataset, labelset, label)
    idx = find(labelset == label);
    idx = idx(1); % just use the first one
    digit = flipud(dataset(:,:,idx));
end

function plotNumber(dataset, labelset, lblArr)
    [~,idx] = size(lblArr);
    word = [];
    for n = lblArr
        word = [word findDigit(dataset,labelset,n)];
    end
    surf(word,'EdgeColor','interp'); view(2);
end