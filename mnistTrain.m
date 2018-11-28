% This function will train the mnistModel with training and
% test data, will output the loss
function [mdl, l] = mnistTrain(trainImg, trainLbl, testImg, testLbl, model)

    % will downsampling help here?
    %trainImg = arrayfun( @(x) subSample(x,1.2), trainImg);
    
    [w,h,d] = size(trainImg);    
    training = reshape( trainImg, [w*h,d])';
    
    [w,h,d] = size(testImg);    
    testing = reshape( testImg, [w*h,d])';
    
    if strcmp(model,'svm')
        
    end
    
    if strcmp(model,'tree')
       mdl = fitctree(training, trainLbl);
       l = [loss(mdl, testing, testLbl); resubLoss(mdl)];
    end
    
    if strcmp(model,'net')
    
        ndata = [training;testing]';
        nlbl  = [trainLbl;testLbl]';

        trainS = length(trainLbl);
        testS  = length(testLbl);
        totalS = trainS + testS;
        trainRatio = trainS/totalS;

        optimize = 0;

        if optimize == 1
            N = 30;
            perf = zeros (1, N);

            for k = 1:N
                [net, l] = fitNet(ndata, nlbl, k, trainRatio);
                perf(k) = l;
            end

            %plot(loss)

            % find the best node combo
            [~, idx] = min(perf);
        end

        idx = 10; % override

        % train the correct net    
        [mdl,l] = fitNet(ndata,nlbl,idx,trainRatio);
    end
    
    if strcmp(model,'knn')
        
       l = zeros(2,10);
        
       for i = 1:10
           mdl = fitcknn(training, trainLbl, 'NumNeighbors',i);
           l(:,i) = [loss(mdl, testing, testLbl); resubLoss(mdl)];
           i
       end
       
       [~, idx] = min(l(1,:));
       mdl = fitcknn(training, trainLbl, 'NumNeighbors',idx);
       l = [loss(mdl, testing, testLbl); resubLoss(mdl)];
       
    end
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

    [net, ~] = train(net, data, lbl);
    ypred = net(data);

    l = perform(net,lbl,ypred);
end