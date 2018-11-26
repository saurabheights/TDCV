close all; clear all;

%% Import needed paths
% Add auxiliary_code to directory
addpath('auxiliary_code/');

% Load the teabox.ply
[vertices,faces] = read_ply('./data/model/teabox.ply');
faces = faces + 1; %Make array 1-indexed.

fig2d = figure('Name', 'Image', 'Color', [0.4 0.6 0.7]);
fig3d = figure('Name', 'SiftIn3D', 'Color', [0.4 0.6 0.7]);
grid on
axis equal
xlabel('X');
ylabel('Y');
zlabel('Z');
xlim([-10,10]./8);
ylim([-10,10]./8);
zlim([-10,10]./8);
hold on; plot3(vertices(:,1),vertices(:,2),vertices(:,3),'r*'); % Plot the vertices of teabox.

%% Careful
%% The vertex index used in ply for a corner, should match the corner index
%% given to that corner when the corner pixel location is manually selected.
texture_dir = './data/images/init_texture/';
filePattern = fullfile(texture_dir, '*.JPG');
jpegFiles = dir(filePattern);
numImages = length(jpegFiles);
imgs = zeros(2456, 3680, 3, numImages);
for k = 1:numImages
    baseFileName = jpegFiles(k).name;
    fullFileName = fullfile(texture_dir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    img = imread(fullFileName);
    imgs(:, :, :, k) = img;
end

%% Load pixel location of the corners
corners_in_images = {[1:4,7:8];[1:4,6:8];[1:4,6:7];[1:4,5:7];[1:6];[1:6,8];[1:5,8];[1:5,7:8]};
load corner_pixel

%% Create intrinsic matrix
focalLength = [2960.37845 2960.37845];
principalPoint = [1841.68855 1235.23369];
imageSize = [2456 3680]; % See - https://www.mathworks.com/help/vision/ref/cameraintrinsics.html
IntrinsicMatrix = [2960.37845,0,0;0,2960.37845,0;1841.68855,1235.23369,1];

% generate the camera parameters
cameraParams = cameraParameters('IntrinsicMatrix', IntrinsicMatrix, 'ImageSize', imageSize);

%% Estimate world camera pose and display camera pose on same plot.
worldOrientations= zeros(8, 3, 3);
worldLocations= zeros(8, 3);
for k = 1:numImages
    imagePoints = corner_pixel{k};
    worldPoints = single(vertices(corners_in_images{k},:));
    [worldOrientation,worldLocation] = estimateWorldCameraPose(imagePoints,worldPoints, cameraParams, 'MaxReprojectionError',6);
    worldOrientations(k, :, :) = worldOrientation;
    worldLocations(k, :, :) = worldLocation;
end

% Download vlfeat binary package from http://www.vlfeat.org/download.html
run('vlfeat-0.9.21-bin/toolbox/vl_setup.m');

%% Compute SIFT features on each image
for k = 1:numImages % length(jpegFiles) %ToDo - remove length
    I = imgs(:, :, :, k);
    
    %% Display Image and compute its sift points.
    figure(fig2d); hold off; % off to remove previous figure content.
    imshow(uint8(I));
    I = single(rgb2gray(uint8(I)));
    f = vl_sift(I) ;
    perm = randperm(size(f,2)) ;
    disp('Truncated sift features to 500 to reduce the computation');
    sel = perm(1:500);
    IndicesInTeaImage = zeros(1,size(sel, 2));
    
    %% Compute the Projection matrix for kth camera(image)
    C = worldLocations(k, :, :);
    [R, t] = cameraPoseToExtrinsics(squeeze(worldOrientations(k, :, :)), C);
    RT = [R; t];
    projectionMatrix = RT * cameraParams.IntrinsicMatrix;
    
    %% For each triangle made of the vertices on the box
    facesFacingCameraIndices = zeros(1,size(faces, 1));
    for faceIndex = 1:size(faces, 1)
        %% Check if the plane made from the vertices normal makes < 90 degrees, else continue
        faceVertices = vertices(faces(faceIndex,:), :);
        faceNormal = cross(faceVertices(1,:) - faceVertices(2,:), ...
            faceVertices(2,:) - faceVertices(3,:));
        faceToCameraVector = worldLocations(k, :) - mean(faceVertices,1);
        
        if dot(faceNormal, faceToCameraVector) > 0
            facesFacingCameraIndices(faceIndex) = 1;
            
            %% Compute location of the face vertices in kth camera coordinate system.
            
            % Below code didnt work. ToDo - Debug this. Convert to
            % homogeneous coordinate system
            %             K = cameraParams.IntrinsicMatrix'; % Matlab
            %             stores transpose of camera intrinsic params Rt =
            %             zeros(3,4); Rt(1:3, 1:3) = R; Rt(:, 4) = t;
            %             pixelLocations = K * (R * faceVertices' + t');
            %             pixelLocations = bsxfun(@rdivide, pixelLocations,
            %             pixelLocations(3,:)); % Divide from last
            %             coordinate pixelLocations(end,:) = []; % make
            %             non-homogeneous
            %
            %             figure(fig2d); x = pixelLocations(1, :)'; y =
            %             imageSize(2) - pixelLocations(2, :)'; hold on;
            %             plot(x, y, 'r*');
            % %             pause(10);
            
            % Same as above code. But matrix multiplcation is done in
            % reverse and with taking transpose.
            faceHomogeneousCoordinates = ones(3, 4); % Assumption: Each face has 3 vertex. Each vertex has 4 homogeneous coordinates
            faceHomogeneousCoordinates(1:3, 1:3) = faceVertices;
            pixelLocations = faceHomogeneousCoordinates * projectionMatrix;
            pixelLocations = bsxfun(@rdivide, pixelLocations(:, 1:2), pixelLocations(:, 3)); % Divide from last coordinate
            
            figure(fig2d);
            x = pixelLocations(:, 1); y = pixelLocations(:, 2);
            hold on; plot(x, y, 'r*');
            
            %% Save all sift feature indices which lie in the projected traingular meshes
            P1 = pixelLocations(1,:);
            P2 = pixelLocations(2,:);
            P3 = pixelLocations(3,:);
            P12 = P1-P2; P23 = P2-P3; P31 = P3-P1;
            s = det([P1-P2;P3-P1]); % https://www.mathworks.com/matlabcentral/answers/277984-check-points-inside-triangle-or-on-edge-with-example
            for siftRandomIndex = 1:size(sel, 2)
                P = f(1:2, sel(siftRandomIndex))';
                flag = sign(det([P31;P23]))*sign(det([P3-P;P23])) >= 0 & ...
                    sign(det([P12;P31]))*sign(det([P1-P;P31])) >= 0 & ...
                    sign(det([P23;P12]))*sign(det([P2-P;P12])) >= 0 ;
                if flag
                    IndicesInTeaImage(siftRandomIndex) = 1;
                end
            end
            
            %% Find indices of sift points and reduce the number to make faster.
            IndicesInTeaImage = find(IndicesInTeaImage == 1);
            numOfIndices = min(size(IndicesInTeaImage,2), 50);
            IndicesInTeaImage = IndicesInTeaImage(1:numOfIndices);
            
            %% Plot sift points on the image.
            figure(fig2d); hold on;
            plot(f(1, sel(IndicesInTeaImage)), f(2, sel(IndicesInTeaImage)), 'y*');
            
            %% Project each sift point to plane of teabox triangle
            % Origin is worldLocation from estimate camera pose.
            orig =repmat(C, numOfIndices, 1);
            imageSiftPoints = [f(1, sel(IndicesInTeaImage)); f(2, sel(IndicesInTeaImage))]';
            worldSiftPoints = pointsToWorld(cameraParams, R, t, imageSiftPoints);
            dir = zeros(numOfIndices, 3); % ToDo: Why the last coordinate is set to zero? It represents z-plane which lie on the bottom of teabox.
            dir(:, 1:2) = worldSiftPoints;
            dir = dir - orig;
            faceVerticesIndex = faces(faceIndex, :);
            
            % Find where ray passing through sift points meets the face.
            vert0 = vertices(faceVerticesIndex(1), :);
            vert1 = vertices(faceVerticesIndex(2), :);
            vert2 = vertices(faceVerticesIndex(3), :);
            [INTERSECT, T, U, V, XCOOR] = TriangleRayIntersection(orig, dir, vert0, vert1, vert2, 'planeType', 'one sided', 'fullReturn', true);
            
            % Display the points.
            figure(fig3d);
            hold on; plot3(orig(:,1),orig(:,2),orig(:,3),'b*');
            hold on; plot3(vert0(:,1),vert0(:,2),vert0(:,3),'b*');
            hold on; plot3(vert1(:,1),vert1(:,2),vert1(:,3),'b*');
            hold on; plot3(vert2(:,1),vert2(:,2),vert2(:,3),'b*');
            hold on; plotCamera('Size',0.1,'Orientation',squeeze(worldOrientations(k, :, :)), 'Location', squeeze(worldLocations(k, :, :)), 'AxesVisible', true);
            hold on; scatter3(XCOOR(find(INTERSECT == 1),1), ...
                XCOOR(find(INTERSECT == 1),2), ...
                XCOOR(find(INTERSECT == 1),3), 2); %
            pause(1);
        end
    end
    pause(1);
end