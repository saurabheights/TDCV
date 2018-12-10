clear all;
close all;

% Load the teabox.ply
addpath('../Exercise01/auxiliary_code/');
[vertices,faces] = read_ply('../Exercise01/data/model/teabox.ply');
verticesHomogeneous = [vertices ones(1, 8)']; % Make homogeneous

figurePlotMatch = figure('Name', 'figurePlotMatch');
fig3d = figure('Name', 'SiftIn3D', 'Color', [0.4 0.6 0.7]); set(gcf,'Visible', 'off');
grid on;axis equal;xlabel('X');ylabel('Y');zlabel('Z');xlim([-10,10]./4);ylim([-10,10]./4);zlim([-10,10]./4);
hold on; plot3(verticesHomogeneous(:,1),verticesHomogeneous(:,2),verticesHomogeneous(:,3),'r*'); % Plot the vertices of teabox.            

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
load('../Exercise02/Exercise02-SiftIn3d.mat', 'SiftIn3d');

%% Read all the files
color_images_dir = './images/tracking/';
% color_images_dir = '../Exercise01/data/images/init_texture/';
filePattern = fullfile(color_images_dir, '*.JPG');
jpegFiles = dir(filePattern);
numImages = length(jpegFiles);
imgs = zeros(2456, 3680, 3, numImages);

% Algorithm Parameters
THRESH = 1.6;
MinimumMatchesToKeep = 500;
CONFIDENCE=99.9;
% Make output directory
OUTDIR = 'Thresh_' + string(THRESH) + '_Matches_' + string(MinimumMatchesToKeep) + '_Confidence_' + string(CONFIDENCE);
mkdir('output', OUTDIR);
OUTDIR = 'output/' + OUTDIR;

figure2d = figure();

%% Detect object in the first image

%% Read the first color image
fullFileName = fullfile(color_images_dir, jpegFiles(1).name);
fprintf(1, 'Now reading %s\n', fullFileName);
img0 = imread(fullFileName);
figure(figure2d);
hold off; imshow(img0);

%% Compute sift feature in image.
[f0, d0] = vl_sift(single(rgb2gray(uint8(img0))));

%% Compute 3d-2d correspondences
[matches, scores] = vl_ubcmatch(d0, SiftIn3d.featuresd, THRESH);
matchesScoreConcat = [matches; scores];
sortedMatches = sortrows(matchesScoreConcat',3)';
if size(sortedMatches, 2) > MinimumMatchesToKeep
    sortedMatches = sortedMatches(1:2, 1:MinimumMatchesToKeep);
    disp('Truncated matching sift pairs to ' + string(MinimumMatchesToKeep) + '  only.');
end

%% Concatenate 3d locations and 2d location of correspondences
% sortedMatches(1,i) stores which descriptor from 2d image is matched.
% sortedMatches(2,i) stores which descriptor from 3d image is matched.
bestImagePoints = f0(1:2, sortedMatches(1,:))' ;
bestWorldPoints = SiftIn3d.threeDLoc(:, sortedMatches(2,:))';

%% Apply ransac and p3p
% estimateCameraWorldPose uses ransac and p3p to compute the best fit
[worldOrientation,worldLocation, inlierIdx] = estimateWorldCameraPose(...
    bestImagePoints, bestWorldPoints, cameraParams, ...
    'MaxReprojectionError',1, 'Confidence', CONFIDENCE, 'MaxNumTrials', 2000);
figure(fig3d);
clf(fig3d);
grid on;axis equal;xlabel('X');ylabel('Y');zlabel('Z');xlim([-10,10]./8);ylim([-10,10]./8);zlim([-10,10]./8);
hold on; plot3(verticesHomogeneous(:,1),verticesHomogeneous(:,2),verticesHomogeneous(:,3),'r*'); % Plot the vertices of teabox.            
hold on; plotCamera('Size',0.1,'Orientation', worldOrientation, ...
    'Location', worldLocation, 'AxesVisible', true);

%% Compute projection matrix
[R0, t0] = cameraPoseToExtrinsics(worldOrientation, worldLocation);
RT0 = [R0; t0];
projectionMatrix = RT0 * cameraParams.IntrinsicMatrix;
pixelLocations = verticesHomogeneous * projectionMatrix;
pixelLocations = bsxfun(@rdivide, pixelLocations(:, 1:2), pixelLocations(:, 3)); % Divide from last coordinate

%% Display box on the image
figure(figure2d);
draw_box(pixelLocations);

%% save output image
F = getframe;
fullFileName = fullfile(OUTDIR, '1.png');
imwrite(F.cdata, char(fullFileName));

%% Exercise 3 - Starts here

%% Filter out the inliers(sift points) used for estimating camera pose
worldOrientation0 = worldOrientation;
worldLocation0 = worldLocation;

%% Main tracking for all images except first
for k = 2:numImages
    %% Read the color image
    baseFileName = jpegFiles(k).name;
    fullFileName = fullfile(color_images_dir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    img1 = imread(fullFileName);
    figure(figure2d);
    hold off; imshow(img1);
    
    %% Compute sift feature in image.
    [f1, d1] = vl_sift(single(rgb2gray(uint8(img1))));
    
    %% Compute where the sift point from previous image meets the 3d box
    %% Only keep features which meet the box.
    [M, mIndices] = Get3DPoints(f0(1:2,:),cameraParams, R0,t0,vertices,faces);
    M = M(mIndices, :);
    fWorld = f0(:,mIndices);
    dWorld = d0(:,mIndices);  
    
    %% Find matches between previous and current image and filter out bad ones.
    [matches, scores] = vl_ubcmatch(dWorld, d1);
    if size(matches, 2) > MinimumMatchesToKeep
        matchesScoreConcat = [matches; scores];
        sortedMatches = sortrows(matchesScoreConcat',3)';
        matches = sortedMatches(1:2, 1:MinimumMatchesToKeep);
        disp('Truncated matching sift pairs to ' + string(MinimumMatchesToKeep) + '  only.');
    end
    displacement = sqrt(sum((fWorld(1:2, matches(1,:)) - f1(1:2, matches(2,:)))'.^2, 2)');
    matches = matches(:, find(displacement < 2 * mean(displacement)));
    
    M = M(matches(1,:), :);
    m = f1(1:2, matches(2,:))';
    
    %% Plot the matches
    plotMatches(figurePlotMatch, img0, img1, fWorld, f1, matches);
    
    %% LM Algo - Compute new trajectory using previous pose as initial point
    %% Output needed: worldOrientation1, worldLocation1
    iterations = 500; % Maximum iterations
    updateThreshold = 1e-6; % Convergence criterion
    [R1, t1] = LM(M, m, R0, t0, cameraParams, iterations, updateThreshold);
    [worldOrientation1, worldLocation1] = cameraPoseToExtrinsics(R1, t1);
    
    %% Plot the camera trajectory
    figure(fig3d);
    clf(fig3d);
    grid on;axis equal;xlabel('X');ylabel('Y');zlabel('Z');xlim([-10,10]./8);ylim([-10,10]./8);zlim([-10,10]./8);
    hold on; plot3(verticesHomogeneous(:,1),verticesHomogeneous(:,2),verticesHomogeneous(:,3),'r*'); % Plot the vertices of teabox.            
    hold on; plotCamera('Size',0.1,'Orientation', worldOrientation1, ...
        'Location', worldLocation1, 'AxesVisible', true);
    
    %% Compute projection matrix
    RT1 = [R1; t1];
    projectionMatrix = RT1 * cameraParams.IntrinsicMatrix;
    pixelLocations = verticesHomogeneous * projectionMatrix;
    pixelLocations = bsxfun(@rdivide, pixelLocations(:, 1:2), pixelLocations(:, 3)); % Divide from last coordinate

    %% Draw the box on the image using the new pose
    figure(figure2d);
    draw_box(pixelLocations);
    F = getframe;
    fullFileName = fullfile(OUTDIR, string(k) + '.png');
    imwrite(F.cdata, char(fullFileName));
    
    %% Set current frame R, t as prev frame's R and t.
    f0 = f1;
    d0 = d1;
    R0 = R1;
    t0 = t1;
    img0 = img1;
end

%% Function to compute reprojection error for all 2d-3d correspondences
function [err] = energy(point2D, point3D, R, T, A)
    point3DHomogeneous = [point3D, ones(size(point3D, 1), 1)];
    projectionMatrix1 = [R; T] * A';
    pixelLocations1 = point3DHomogeneous * projectionMatrix1;
    pixelLocations1 = bsxfun(@rdivide, pixelLocations1(:, 1:2), pixelLocations1(:, 3));
    err = sqrt(sum((pixelLocations1 - point2D).^2,2));
%     sizeP = size(point2D,1);
%     err = zeros(sizeP,1);
%     projectionMatrix = A*[R, T'];
%     for i = 1:sizeP(1)
%         proj2D = projectionMatrix * [point3D(i,:),1]';
%         proj2D = proj2D/proj2D(3);
%         err(i) = (point2D(i,1) - proj2D(1))^2 + (point2D(i,2) - proj2D(2))^2;
%     end
end

%% Function to compute residual for all 2d-3d correspondences
function [res] = residual(point2D, point3D, R, T, A)
    point3DHomogeneous = [point3D, ones(size(point3D, 1), 1)];
    projectionMatrix1 = [R; T] * A';
    pixelLocations1 = point3DHomogeneous * projectionMatrix1;
    pixelLocations1 = bsxfun(@rdivide, pixelLocations1(:, 1:2), pixelLocations1(:, 3));
    res = reshape((pixelLocations1 - point2D)',1,[])';
    
%     sizeP = size(point2D, 1);
%     res = zeros(sizeP*2,1);
%     projectionMatrix = A*[R, T'];    
%     for i = 1:sizeP
%         proj2D = projectionMatrix * [point3D(i,:),1]';
%         proj2D = proj2D/proj2D(3);
%         res(2*i-1:2*i) = [( proj2D(1) - point2D(i,1) ); ( proj2D(2) - point2D(i,2) )];
%     end
end

function [Rnew, Tnew] = GradientDescent(M, m, R0, t0, cameraParams, iterations, updateThreshold)
    rotation = rotationMatrixToVector(R0);
    res = [rotation, t0];
    lambda = 0.00000001;
    update = updateThreshold + 1; % Any value higher than convergence criterion
    iter = 1;
    Rnew = R0;
    Tnew = t0;
    A = cameraParams.IntrinsicMatrix';
    while iter <= iterations && update > updateThreshold
        % Compute Jacobian
        J = jacobian(M, Rnew, res(1:3), Tnew, A);
        
        % Compute Residual
        resid = residual(m, M, Rnew, Tnew, A);
        
        % Compute Update
        delta = - (2*lambda*J'*resid)';
        res_new = res + delta;
        e = energy(m, M, Rnew, Tnew, A);
        error_new = energy(m, M, rotationVectorToMatrix(res_new(1:3)), res_new(4:6), A);
        
        if abs(sum(error_new)) > abs(sum(e))
            lambda = lambda/2;
        else
            res = res_new;
            Rnew = rotationVectorToMatrix(res(1:3));
            Tnew = res(4:6);
            update = norm(delta);
            iter = iter + 1;
        end 
        fprintf('%d | %.8f | %.8f\n', iter, abs(sum(e)), abs(sum(error_new)));
    end
end

function [Rnew, Tnew] = LM(M, m, R0, t0, cameraParams, iterations, updateThreshold)
    rotation = rotationMatrixToVector(R0);
    res = [rotation, t0];
    lambda = 0.00001;
    update = updateThreshold + 1; % Any value higher than convergence criterion
    iter = 1;
    Rnew = R0;
    Tnew = t0;
    A = cameraParams.IntrinsicMatrix';
    while iter <= iterations && update > updateThreshold
        % Compute Jacobian
        J = jacobian(M, Rnew, res(1:3), Tnew, A);
        
        % Compute Residual
        resid = residual(m, M, Rnew, Tnew, A);
        
        % Compute Update
        delta = - (2*lambda*J'*resid)';
        delta = -inv(J' * J + lambda*eye(6)) * J' * resid;
        res_new = res + delta';
        e = energy(m, M, Rnew, Tnew, A);
        error_new = energy(m, M, rotationVectorToMatrix(res_new(1:3)), res_new(4:6), A);
        if abs(sum(error_new)) > abs(sum(e))
            lambda = 10 * lambda;
        else
            lambda = lambda / 10;
            res = res_new;
            Rnew = rotationVectorToMatrix(res(1:3));
            Tnew = res(4:6);
            update = norm(delta);
            iter = iter + 1;
        end
    fprintf('%d | %.8f | %.8f\n', iter, abs(sum(e)), abs(sum(error_new)));
    end
end

function [Jac] = jacobian(point3D, R, Rexp, T, A)
    Jac = [];
    sizePoints = size(point3D);
    
    for j = 1:sizePoints(1)
        current3Dpoint = point3D(j,:);
        proj2D = A*[R, T']*[current3Dpoint,1]';

        U = proj2D(1);
        V = proj2D(2);
        W = proj2D(3);
        der1 = [1/W  0  -U/(W^2); 0  1/W  -V/(W^2)];

        der2a = rodriguesDerivative(Rexp, R, 1)*current3Dpoint';
        der2b = rodriguesDerivative(Rexp, R, 2)*current3Dpoint';
        der2c = rodriguesDerivative(Rexp, R, 3)*current3Dpoint';

        Jtmp = der1*A*[der2a der2b der2c eye(3)];
        Jac = [Jac; Jtmp];
    end
end

function [val] = rodriguesDerivative(v, R, i)
    basis = [0;0;0];
    basis(i) = basis(i) + 1;
    
    tmp = cross(v, ((eye(3)-R)*basis));
    val = (v(i) * skew(v) + skew(tmp))*R;
    val = val / (norm(v)^2);
end

function [rodr] = skew(v)
    rodr = [0, -v(3), v(2); v(3), 0, -v(1); -v(2) v(1) 0];
end

function draw_box(pixelLocations)
% Draws the vertices and edges of teabox on the image
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
end

function [] = plotMatches(figurePlotMatch, Ia, Ib, fa, fb, matches)
    figure(figurePlotMatch) ; clf ;
    imagesc(cat(2, rgb2gray(int64(Ia)), rgb2gray(int64(Ib)))) ;

    xa = fa(1,matches(1,:)) ;
    xb = fb(1,matches(2,:)) + size(Ia,2) ;
    ya = fa(2,matches(1,:)) ;
    yb = fb(2,matches(2,:)) ;

    hold on ;
    h = line([xa ; xb], [ya ; yb]) ;
    set(h,'linewidth', 0.00001, 'color', 'b') ;

    vl_plotframe(fa(:,matches(1,:))) ;
    fb(1,:) = fb(1,:) + size(Ia,2) ;
    vl_plotframe(fb(:,matches(2,:))) ;
    axis image off ;
end