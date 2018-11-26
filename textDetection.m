%Detect regions in a image that contain text

Image = imread('asdflkjalfja.jpg'); %need to research how to implement image processing in real time
I = rgb2gray(Image); 

%Pt.1: Detect Candadite Text Region using MSER
[mserRegions,mserConnComp] = detectMSERFeatures(I,'RegionAreaRange', [200,8000], 'ThresholdDelta', 4);
figure; 
imshow(I);
hold on; 
plot(mserRegions, 'showPixelList', true,'showEllipses',False)
title('MSER regions');
hold off; 

%Part 2: 
mserStats = regionprops(mserConnComp, 'BoundingBox', 'Eccentricity','Solidity','Extent','Euler','Image'); 
bbox = vertcat(mserStats.BoundingBox);
w = bbox(:,3);
h = bbox(:,4);
aspectRation = w./h; 
filterIdx  = aspectRatio' > 3; 
filterIdx = filterIdx | [mserStats.Eccentricity] > .995; 
filterIdx = filterIdx | [mserStats.Solidarity] < .3;
filterIdx = filterIdx | [mserStats.Extent] < 0.2 | [mserStats.Extent] > 0.9;
filterIdx = filterIdx | [mserStats.EulerNumber] < -4; 
mserStats(filterIdx) = []; 
mserRegions(filterIdx) = []; 
figure; 
imshow(I); 
hold on; 
plot(mserRegions, 'showPixelList', true,'showEllipses',false);
title('After Removing Non-Text Regions Based On Geometric Properties');
hold off;  
