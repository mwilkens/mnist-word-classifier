% image to classify
image = "testImage.png";

% Files for MNIST
TESTIMG_FILE = "t10k-images-idx3-ubyte";
TESTLBL_FILE = "t10k-labels-idx1-ubyte";
TRAINIMG_FILE = "train-images-idx3-ubyte";
TRAINLBL_FILE = "train-labels-idx1-ubyte";

[mnistTestImg, mnistTestLbl] = mnistParse(TESTIMG_FILE, TESTLBL_FILE, 25, 0);

plotLetter(mnistTestImg,4);

function plotLetter(arr, ltrIdx)
% plotLetter - plots a letter of the MNIST training set
    % plot the 3d array
    surf(arr(:,:,ltrIdx), 'EdgeColor','interp');
    % look at the overview so it looks like a 2d image
    view(2)
end