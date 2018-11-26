% This file will classify a 50x50 normalized 2D array into a text letter
% will also output an uncertainty loss
function [ltr] = mnistClassify(net, array)
    [w,h,d] = size(array);
    if d ~= 1
        % we should only have 1 image!
        ltr = -1;
        return
    end
    
    img = reshape(array,[w*h, 1]);
    
    ltr = round(net(img));
end