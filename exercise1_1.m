close all; clear all;

%% Import needed paths
% Add auxiliary_code to directory
addpath('auxiliary_code/');

% Load the teabox.ply
[vertices,faces] = read_ply('./data/model/teabox.ply');
faces = faces + 1;

fig1 = figure('Name', 'CameraAndObjectPose', 'Color', [0.4 0.6 0.7]);
% pcshow only displays 4 points. pcshow(pointCloud(vertex), 'VerticalAxis',
% 'Z', ...
%     'VerticalAxisDir', 'Up');
plot3(vertices(:,1),vertices(:,2),vertices(:,3),'g*');
grid on
axis equal
xlabel('X');
ylabel('Y');
zlabel('Z');
xlim([-10,10]./4);
ylim([-10,10]./4);
zlim([-10,10]./4);
hold on;
hold off;

%% Careful
%% The vertex index used in ply for a corner, should match the corner index
%% given to that corner when the corner pixel location is manually selected.

texture_dir = './data/images/init_texture/';
filePattern = fullfile(texture_dir, '*.JPG');
jpegFiles = dir(filePattern);
numImages = 1; % length(jpegFiles);
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
cameraParams = cameraIntrinsics(focalLength, principalPoint, imageSize)

%% Estimate world camera pose and display camera pose on same plot.
worldOrientations= zeros(8, 3, 3);
worldLocations= zeros(8, 3);
for k = 1:numImages
    imagePoints = corner_pixel{k};
    worldPoints = model.Location(corners_in_images{k},:);
    [worldOrientation,worldLocation] = estimateWorldCameraPose(imagePoints,worldPoints, cameraParams, 'MaxReprojectionError',5);
    worldOrientations(k, :, :) = worldOrientation;
    worldLocations(k, :, :) = worldLocation;
    hold on;
    plotCamera('Size',0.1,'Orientation',worldOrientation, ...
        'Location', worldLocation);
    hold off;
end

% Download vlfeat binary package from http://www.vlfeat.org/download.html
run('vlfeat-0.9.21-bin/toolbox/vl_setup.m');

fig2 = figure('Name', 'Image', 'Color', [0.4 0.6 0.7]);
%% Compute SIFT features on each image
for k = 1:numImages % length(jpegFiles) %ToDo - remove length
    I = imgs(:, :, :, k);
    
    imshow(uint8(I));
    I = single(rgb2gray(uint8(I)));
    [f,d] = vl_sift(I) ;
    perm = randperm(size(f,2)) ;
    sel = perm(1:50) ;
    %     h1 = vl_plotframe(f(:,sel)) ; set(h1,'color','r','linewidth',3) ;
    %     h3 = vl_plotsiftdescriptor(d(:,sel),f(:,sel)) ;
    %     set(h3,'color','g') ;
    %% Compute the Projection matrix for kth camera(image)
    C = worldLocations(k, :, :);
    [R, t] = cameraPoseToExtrinsics(squeeze(worldOrientations(k, :, :)), C);
    K = cameraParams.IntrinsicMatrix'; % Matlab stores transpose of camera intrinsic params
    Rt = zeros(3,4);
    Rt(1:3, 1:3) = R;
    Rt(:, 4) = t;
    
    %% For each triangle made of the vertices on the box
    for faceIndex = 1:size(faces, 1)
        %% Check if the plane made from the vertices normal makes < 90 degrees, else continue
        faceVertices = vertices(faces(faceIndex,:), :);
        faceNormal = cross(faceVertices(1,:) - faceVertices(2,:), ...
            faceVertices(2,:) - faceVertices(3,:));
        faceToCameraVector = worldLocations(k, :) - ((faceVertices(1,:) + faceVertices(2,:) + ...
            faceVertices(3,:)) / 3);
        
        if dot(faceNormal, faceToCameraVector) > 0
            %% Debug if the faces detected are correct are not. Run this code only for one camera
            figure(fig1);
            hold on;
            fill3(faceVertices(:,1),faceVertices(:,2),faceVertices(:,3), [0 , 0, 0.5]);

            %% Compute location of the face vertices in kth camera coordinate system.
            
            % Convert to homogeneous coordinate system
            faceHomogeneousCoordinates = ones(4, 3); % Assumption: Each face has 3 vertex
            faceHomogeneousCoordinates(1:3, 1:3) = faceVertices';
            pixelLocations = K * Rt * faceHomogeneousCoordinates;
            pixelLocations = bsxfun(@rdivide, pixelLocations, pixelLocations(3,:)); % Divide from last coordinate
            pixelLocations(end,:) = []; % make non-homogeneous
            
            figure(fig2);
            x = pixelLocations(1, :)'; y = imageSize(2) - pixelLocations(2, :)';
            hold on; plot(x, y, 'r*');
            pause(10);
            %% 

            %% Save all sift feature indices which lie in the projected traingular meshes

            %% Project each sift point to plane of teabox triangle
%             TriangleRayIntersection('planeType', 'one sided', 'fullReturn', true);
            %% Reject the points which did not intersect with the teabox triangle

        end
    end

end
pause(10);