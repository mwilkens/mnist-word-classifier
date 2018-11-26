% This file will classify a 50x50 normalized 2D array into a text letter
% will also output an uncertainty loss
function [ltr,l] = mnistClassify(mdl, array)
    [w,h,d] = size(array);
    if d ~= 1
        % we should only have 1 image!
        ltr = -1;
        return
    end
    
    img = reshape(array,[w*h, 1]);
    
    net = 0;
    
    if net == 1
        ltr = round(mdl(img));
        l=0;
    end
    
    if net == 0
        [ltr,l] = predict(mdl,img');
    end
end