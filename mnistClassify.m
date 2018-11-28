% This file will classify a 50x50 normalized 2D array into a text letter
% will also output an uncertainty loss
function [ltr,l] = mnistClassify(mdl, array, modelType)
    [w,h,d] = size(array);
    if d ~= 1
        % we should only have 1 image!
        ltr = -1;
        return
    end
    
    img = reshape(array,[w*h, 1]);
    
    if strcmp(modelType,'svm')
        [ltr,l] = predict(mdl,img');
    end
    
    if strcmp(modelType,'tree')
        [ltr,l] = predict(mdl,img');
    end
    
    if strcmp(modelType,'net')
        ltr = round(mdl(img));
        l=0;
    end
    
    if strcmp(modelType,'knn')
        [ltr,l] = predict(mdl,img');
    end
end