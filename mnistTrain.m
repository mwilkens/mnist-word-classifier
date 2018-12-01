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
        tmp = templateSVM('KernelFunction','Polynomial',...
                            'PolynomialOrder', 4, 'KernelScale','auto',...
                            'RemoveDuplicates', true, 'Solver', 'SMO',...
                            'Verbose', 1);
        mdl = fitcecoc(training,trainLbl,'Learners', tmp);
        l = [loss(mdl, testing, testLbl); resubLoss(mdl)];
    end
    
    if strcmp(model,'tree')
       mdl = fitctree(training, trainLbl);
       l = [loss(mdl, testing, testLbl); resubLoss(mdl)];
    end
    
    if strcmp(model,'net')
    
        ndata = [training;testing]';
        nlbl  = dummyvar(categorical([trainLbl;testLbl]));

        trainS = length(trainLbl);
        testS  = length(testLbl);
        totalS = trainS + testS;
        trainRatio = trainS/totalS;

        optimize = 0;

        if optimize == 1
            N = 30;
            LYRS = 10;
            perf = zeros (N,LYRS);

            for ly = 1:LYRS
                for k = 1:N
                    [net, l] = fitNet(ndata, nlbl', repmat(k,1,ly), trainRatio);
                    perf(k,ly) = l;
                end
            end

            figure
            surf(perf);

            % find the best node combo
            minPerf = min(perf(:));
            [n,l] = find(perf==minPerf);
            idx = repmat(n,1,l);
        end

        if optimize == 0
            idx = [30 30]; % override
        end

        % train the correct net 
        ls = zeros(1,10);
        for k=1:10
            [mdls{k},ls(k)] = fitNet(ndata,nlbl',idx,trainRatio);
        end
        [~, idx] = min(ls);
        mdl = mdls{idx};
        l   = ls(idx);
    end
    
    if strcmp(model,'knn')
        
       l = zeros(2,10);
        
       for i = 1:10
           mdl{i} = fitcknn(training, trainLbl, 'NumNeighbors',i);
           l(:,i) = [loss(mdl{i}, testing, testLbl); resubLoss(mdl{i})];
       end
       
       [~, idx] = min(l(1,:));
       mdl = mdl{idx};
       l = [l(1,idx) l(2,idx)];
       
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