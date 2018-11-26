% This function will train the mnistModel with training and
% test data, will output the loss
function [net, l] = mnistTrain(trainImg, trainLbl, testImg, testLbl)

    % will downsampling help here?
    %trainImg = arrayfun( @(x) subSample(x,1.2), trainImg);
    
    [w,h,d] = size(trainImg);    
    training = reshape( trainImg, [w*h,d])';
    
    [w,h,d] = size(testImg);    
    testing = reshape( testImg, [w*h,d])';
    
    ndata = [training;testing]';
    nlbl  = [trainLbl;testLbl]';
    
    trainS = length(trainLbl);
    testS  = length(testLbl);
    totalS = trainS + testS;
    trainRatio = trainS/totalS;
    
    optimize = 0;
    
    if optimize == 1
        N = 30;
        loss = zeros (1, N);

        for k = 1:N
            [net, l] = fitNet(ndata, nlbl, k, trainRatio);
            loss(k) = l;
        end

        %plot(loss)

        % find the best node combo
        [~, idx] = min(loss);
    end
    
    idx = 10; % override
    
    % train the correct net    
    [net,l] = fitNet(ndata,nlbl,idx,trainRatio);
end

function [img] = subSample(fullimg)
% will subscale an image
    img = fullimg(1:factor:end,1:factor:end);
end

function [net, l] = fitNet(data,lbl,nodes,tr)
    net = patternnet(nodes);

    net.divideParam.trainRatio = tr;
    net.divideParam.valRatio   = 1-tr;
    net.divideParam.trainRatio = 1-tr;

    net.layers{1}.transferFcn = 'tansig';
    net.trainFcn = 'trainrp';

    [net, tr] = train(net, data, lbl);
    ypred = net(data);

    l = perform(net,lbl,ypred);
end