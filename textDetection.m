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
        imshow(image); 
        hold on; 
        plot(mserRegions, 'showPixelList', true,'showEllipses',false);
        title('After Removing Non-Text Regions Based On Geometry');
        hold off;  
    end

    %Part 3: Remove Non-Text Regions Based on Stroke Width Variation
    % Get a binary image of the a region, and pad it to avoid boundary effects
	% during the stroke width computation.
	regionImage = mserStats(6).Image;
	regionImage = padarray(regionImage, [1 1]);

	% Compute the stroke width image.
	distanceImage = bwdist(~regionImage); 
	skeletonImage = bwmorph(regionImage, 'thin', inf);

	strokeWidthImage = distanceImage;
	strokeWidthImage(~skeletonImage) = 0;

    if plot == 1
        % Show the region image alongside the stroke width image. 
        figure
        subplot(1,2,1)
        imagesc(regionImage)
        title('Region Image')

        subplot(1,2,2)
        imagesc(strokeWidthImage)
        title('Stroke Width Image')
    end
	
	% Compute the stroke width variation metric 
	strokeWidthValues = distanceImage(skeletonImage);   
	strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues);
	
	% Threshold the stroke width variation metric
	strokeWidthThreshold = 0.4;
	strokeWidthFilterIdx = strokeWidthMetric > strokeWidthThreshold;

	% Process the remaining regions
	for j = 1:numel(mserStats)
    
    		regionImage = mserStats(j).Image;
    		regionImage = padarray(regionImage, [1 1], 0);
    
    		distanceImage = bwdist(~regionImage);
    		skeletonImage = bwmorph(regionImage, 'thin', inf);
    
    		strokeWidthValues = distanceImage(skeletonImage);
    
    		strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues);
    
    		strokeWidthFilterIdx(j) = strokeWidthMetric > strokeWidthThreshold;
    
	end

	% Remove regions based on the stroke width variation
	mserRegions(strokeWidthFilterIdx) = [];
	mserStats(strokeWidthFilterIdx) = [];

    if plot == 1
        % Show remaining regions
        figure
        imshow(I)
        hold on
        plot(mserRegions, 'showPixelList', true,'showEllipses',false)
        title('After Removing Non-Text Regions Based On Stroke Width Variation')
        hold off
    end

	%Part 4: 
	% Get bounding boxes for all the regions
    bboxes = vertcat(mserStats.BoundingBox);

    % Convert from the [x y width height] bounding box format to the [xmin ymin
    % xmax ymax] format for convenience.
    xmin = bboxes(:,1);
    ymin = bboxes(:,2);
    xmax = xmin + bboxes(:,3) - 1;
    ymax = ymin + bboxes(:,4) - 1;

    % Expand the bounding boxes by a small amount.
    expansionAmount = 0.02;
    xmin = (1-expansionAmount) * xmin;
    ymin = (1-expansionAmount) * ymin;
    xmax = (1+expansionAmount) * xmax;
    ymax = (1+expansionAmount) * ymax;

    % Clip the bounding boxes to be within the image bounds
    xmin = max(xmin, 1);
    ymin = max(ymin, 1);
    xmax = min(xmax, size(image,2));
    ymax = min(ymax, size(image,1));

    % Show the expanded bounding boxes
    expandedBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];
    IExpandedBBoxes = insertShape(image,'Rectangle',expandedBBoxes,'LineWidth',3);

    if plot == 1
        figure
        imshow(IExpandedBBoxes)
        title('Expanded Bounding Boxes Text')
    end

    % Compute the overlap ratio
    overlapRatio = bboxOverlapRatio(expandedBBoxes, expandedBBoxes);

    % Set the overlap ratio between a bounding box and itself to zero to
    % simplify the graph representation.
    n = size(overlapRatio,1); 
    overlapRatio(1:n+1:n^2) = 0;

    % Create the graph
    g = graph(overlapRatio);

    % Find the connected text regions within the graph
    componentIndices = conncomp(g);

    % Merge the boxes based on the minimum and maximum dimensions.
    xmin = accumarray(componentIndices', xmin, [], @min);
    ymin = accumarray(componentIndices', ymin, [], @min);
    xmax = accumarray(componentIndices', xmax, [], @max);
    ymax = accumarray(componentIndices', ymax, [], @max);

    % Compose the merged bounding boxes using the [x y width height] format.
    textBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];

    % Remove bounding boxes that only contain one text region
    numRegionsInGroup = histcounts(componentIndices);
    textBBoxes(numRegionsInGroup == 1, :) = [];

    % Show the final text detection result.
    ITextRegion = insertShape(image, 'Rectangle', textBBoxes,'LineWidth',3);

    if plot == 1
        figure
        imshow(ITextRegion)
        title('Detected Text')
    end

    [numletters, ~] = size(textBBoxes);
    
    % create template array for letters
    ltrs = zeros(20,20,numletters);
    
    % for each letter we need to:
    % 1. make the image square
    % 2. center the image
    % 3. resize it to 20x20 pixels
    for i = 1:numletters
        % for each letter detected
        edgeCoord = round(textBBoxes(i,1:2)); % find the edge coord
        dim = round(textBBoxes(i,3:4));     % find the width and height
        
        edgeCoord2 = edgeCoord + dim;
        
        % this is the array of pixels x:y that we're extracting
        pixel_box1 = edgeCoord(1):edgeCoord2(1);
        pixel_box2 = edgeCoord(2):(edgeCoord2(2)-1);
        
        % find the box dimention
        boxd = max(dim);
        
        % find where to center our image
        center = round(linspace((boxd/2) - (min(dim)/2), ...
                  (boxd/2) + (min(dim)/2), length(pixel_box1)));
              
        % extract the image and place in center of square image
        imgLarge = zeros(boxd, boxd);
        imgLarge(center,1:length(pixel_box2)) = image(pixel_box2,pixel_box1)';
        
        % resize to 20x20 pixels
        ltrs(:,:,i) = imresize(imgLarge', [20,20]);
    end
end
