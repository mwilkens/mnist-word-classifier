%Detect regions in a image that contain text
function [ltrs] = textDetection(image)
% TEXTDETECTION finds objects in image and scales them for classification

    plot = 0;

    %Pt.1: Detect Candadite Text Region using MSER
    [mserRegions,mserConnComp] = detectMSERFeatures(image, ...
                'RegionAreaRange', [200,8000], 'ThresholdDelta', 4);
    
    if plot == 1
        figure; 
        imshow(image);
        hold on; 
        plot(mserRegions, 'showPixelList', true,'showEllipses',False)
        title('MSER regions');
        hold off; 
    end

    %Part 2: 
    mserStats = regionprops(mserConnComp, 'BoundingBox', ...
                'Eccentricity','Solidity','Extent','Euler','Image'); 
    bbox = vertcat(mserStats.BoundingBox);
    w = bbox(:,3);
    h = bbox(:,4);
    aspectRatio = w./h; 
    filterIdx  = aspectRatio' > 3; 
    filterIdx = filterIdx | [mserStats.Eccentricity] > .995; 
    filterIdx = filterIdx | [mserStats.Solidity] < .3;
    filterIdx = filterIdx | [mserStats.Extent] < 0.2 | ...
                                            [mserStats.Extent] > 0.9;
    filterIdx = filterIdx | [mserStats.EulerNumber] < -4; 
    mserStats(filterIdx) = []; 
    mserRegions(filterIdx) = []; 
    
    if plot == 1
        figure; 
        imshow(I); 
        hold on; 
        plot(mserRegions, 'showPixelList', true,'showEllipses',false);
        title('After Removing Non-Text Regions Based On Geometry');
        hold off;  
    end

    ltrs = mserRegions;
end