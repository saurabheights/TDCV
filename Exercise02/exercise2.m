clear all;
close all;

% Load the teabox.ply
addpath('../Exercise01/auxiliary_code/');
[vertices,faces] = read_ply('../Exercise01/data/model/teabox.ply');
vertices = [vertices ones(1, 8)']; % Make homogeneous

fig3d = figure('Name', 'SiftIn3D', 'Color', [0.4 0.6 0.7]); set(gcf,'Visible', 'off');
grid on;axis equal;xlabel('X');ylabel('Y');zlabel('Z');xlim([-10,10]./8);ylim([-10,10]./8);zlim([-10,10]./8);
hold on; plot3(vertices(:,1),vertices(:,2),vertices(:,3),'r*'); % Plot the vertices of teabox.            

% Download vlfeat binary package from http://www.vlfeat.org/download.html
run('../vlfeat-0.9.21-bin/toolbox/vl_setup.m');

%% Create intrinsic matrix
focalLength = [2960.37845 2960.37845];
principalPoint = [1841.68855 1235.23369];
imageSize = [2456 3680]; % See - https://www.mathworks.com/help/vision/ref/cameraintrinsics.html
IntrinsicMatrix = [2960.37845,0,0;0,2960.37845,0;1841.68855,1235.23369,1];

% generate the camera parameters
cameraParams = cameraParameters('IntrinsicMatrix', IntrinsicMatrix, 'ImageSize', imageSize);

%% Read SIFT descriptor and their 3d location on teabox 3d model
load('Exercise02-SiftIn3d.mat', 'SiftIn3d');

%% Read all the files
color_images_dir = './data/images/detection/';
% color_images_dir = '../Exercise01/data/images/init_texture/';
filePattern = fullfile(color_images_dir, '*.JPG');
jpegFiles = dir(filePattern);
numImages = length(jpegFiles);
imgs = zeros(2456, 3680, 3, numImages);

% Algorithm Parameters
THRESH = 1.6;
MinimumMatchesToKeep = 400;
CONFIDENCE=99.9;
% Make output directory
OUTDIR = 'Thresh_' + string(THRESH) + '_Matches_' + string(MinimumMatchesToKeep) + '_Confidence_' + string(CONFIDENCE);
mkdir('output', OUTDIR);
OUTDIR = 'output/' + OUTDIR;

f1 = figure(); set(gcf,'Visible', 'off');
for k = 1:numImages
    %% Read the color image
    baseFileName = jpegFiles(k).name;
    fullFileName = fullfile(color_images_dir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    img = imread(fullFileName);
    figure(f1);
    hold off; imshow(img);
    
    %% Compute sift feature in image.
    I = single(rgb2gray(uint8(img)));
    [f, d] = vl_sift(I);
    
    %% Compute 3d-2d correspondences
    [matches, scores] = vl_ubcmatch(d, SiftIn3d.featuresd, THRESH);
    matchesScoreConcat = [matches; scores];
    sortedMatches = sortrows(matchesScoreConcat',3)';
    if size(sortedMatches, 2) > MinimumMatchesToKeep
        sortedMatches = sortedMatches(1:2, 1:MinimumMatchesToKeep);
        disp('Truncated matching sift pairs to ' + string(MinimumMatchesToKeep) + '  only.');
    end
    
    %% Concatenate 3d locations and 2d location of correspondences
    % sortedMatches(1,i) stores which descriptor from 2d image is matched.
    % sortedMatches(2,i) stores which descriptor from 3d image is matched.
    bestImagePoints = f(1:2, sortedMatches(1,:))' ;
    bestWorldPoints = SiftIn3d.threeDLoc(:, sortedMatches(2,:))';
    
    %% Apply ransac and p3p
    % estimateCameraWorldPose uses ransac and p3p to compute the best fit
    [worldOrientation,worldLocation, inlierIdx] = estimateWorldCameraPose(...
        bestImagePoints, bestWorldPoints, cameraParams, ...
        'MaxReprojectionError',1, 'Confidence', CONFIDENCE);
    figure(fig3d);
    clf(fig3d);
    grid on;axis equal;xlabel('X');;ylabel('Y');zlabel('Z');xlim([-10,10]./8);ylim([-10,10]./8);zlim([-10,10]./8);
    hold on; plot3(vertices(:,1),vertices(:,2),vertices(:,3),'r*'); % Plot the vertices of teabox.            
    hold on; plotCamera('Size',0.1,'Orientation', worldOrientation, ...
        'Location', worldLocation, 'AxesVisible', true);
    
%     for i=1:size(inlierIdx,1)
%         if inlierIdx(i)
%             figure(fig3d);
%             hold on; plot3(bestWorldPoints(i,1), bestWorldPoints(i,2), bestWorldPoints(i,3),'g*');
%             figure(f1);
%             hold on; plot(bestImagePoints(i, 1), bestImagePoints(i, 2), 'g*');
%             pause(4);
%         end
%     end
    
    %% Compute projection matrix
    [R, t] = cameraPoseToExtrinsics(worldOrientation, worldLocation);
    RT = [R; t];
    projectionMatrix = RT * cameraParams.IntrinsicMatrix;
    pixelLocations = vertices * projectionMatrix;
    pixelLocations = bsxfun(@rdivide, pixelLocations(:, 1:2), pixelLocations(:, 3)); % Divide from last coordinate

    figure(f1);
    x = pixelLocations(:, 1); y = pixelLocations(:, 2);
    hold on; plot(x, y, 'r*');
    % Draw the box edges.
    hold on; line(pixelLocations(1:2, 1), pixelLocations(1:2, 2));
    hold on; line(pixelLocations(2:3, 1), pixelLocations(2:3, 2));
    hold on; line(pixelLocations(3:4, 1), pixelLocations(3:4, 2));
    hold on; line(pixelLocations([1,4], 1), pixelLocations([1,4], 2));
    hold on; line(pixelLocations([5,6], 1), pixelLocations([5,6], 2));
    hold on; line(pixelLocations([6,7], 1), pixelLocations([6,7], 2));
    hold on; line(pixelLocations([7,8], 1), pixelLocations([7,8], 2));
    hold on; line(pixelLocations([5,8], 1), pixelLocations([5,8], 2));
    hold on; line(pixelLocations([1,5], 1), pixelLocations([1,5], 2));
    hold on; line(pixelLocations([2,6], 1), pixelLocations([2,6], 2));
    hold on; line(pixelLocations([3,7], 1), pixelLocations([3,7], 2));
    hold on; line(pixelLocations([4,8], 1), pixelLocations([4,8], 2));
    hold off;
    F = getframe;
    fullFileName = fullfile(OUTDIR, string(k) + '.png');
    imwrite(F.cdata, char(fullFileName));
end
