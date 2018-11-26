% This function will train the mnistModel with training and
% test data, will output the loss
function [mdl, l] = mnistTrain(trainImg, trainLbl, testImg, testLbl)

    % will downsampling help here?
    %trainImg = arrayfun( @(x) subSample(x,1.2), trainImg);
    
    [w,h,d] = size(trainImg);
    
    training = reshape( trainImg, [w*h,d])';
    
    template = templateSVM('KernelFunction','polynomial', ...
        'PolynomialOrder', 4);
    
    mdl = fitcecoc(training,trainLbl,'Learners', template);
    
    [w,h,d] = size(testImg);
    testing = reshape( testImg, [w*h, d])';
    l = loss(mdl,testing,testLbl);   
end

function [img] = subSample(fullimg)
% will subscale an image
    img = fullimg(1:factor:end,1:factor:end);
end
