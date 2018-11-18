close all;

%% Import needed paths
% Add auxiliary_code to directory
addpath('auxiliary_code/');

% Load the teabox.ply
[vertices,faces] = read_ply('./data/model/teabox.ply');
faces = faces + 1;

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
numImages = length(jpegFiles);
imgs = zeros(2456, 3680, 3, numImages);
% At most we will need 8 corners for each image.
imgsCornerLoc = cell(numImages, 8);
for k = 1:length(jpegFiles)
    baseFileName = jpegFiles(k).name;
    fullFileName = fullfile(texture_dir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    img = imread(fullFileName);
    imgs(:, :, :, k) = img;
end

%% Load pixel location of the corners
num_corner = {[1:4,7:8];[1:4,6:8];[1:4,6:7];[1:4,5:7];[1:6];[1:6,8];[1:5,8];[1:5,7:8]};
load corner_pixel


%% Create intrinsic matrix
focalLength = [2960.37845 2960.37845];
principalPoint = [1841.68855 1235.23369];
imageSize = [2456 3680];
cameraParams = cameraIntrinsics(focalLength, principalPoint, imageSize);


%% Estimate world camera pose and display camera pose on same plot.
for k = 1:length(jpegFiles)
    try
        imagePoints = corner_pixel{k};
        worldPoints = model.Location(num_corner{k},:);
        [worldOrientation,worldLocation] = estimateWorldCameraPose(imagePoints,worldPoints, cameraParams, 'MaxReprojectionError',5);
        hold on;
        plotCamera('Size',0.1,'Orientation',worldOrientation, ...
            'Location', worldLocation);
        hold off;
    catch
        fprintf('estimateWorldPose failed - %s, skipped.\n', num2str(k));
    end
end

hold off

% Download vlfeat binary package from http://www.vlfeat.org/download.html
run('vlfeat-0.9.21-bin/toolbox/vl_setup.m');

%% Compute SIFT features on each image
for k = 1:1 % length(jpegFiles) %ToDo - remove length
    I = imgs(:, :, :, k);
    
    image(uint8(I)) ;
    I = single(rgb2gray(uint8(I)));
    [f,d] = vl_sift(I) ;
    perm = randperm(size(f,2)) ;
    sel = perm(1:50) ;
    %     h1 = vl_plotframe(f(:,sel)) ; set(h1,'color','r','linewidth',3) ;
    %     h3 = vl_plotsiftdescriptor(d(:,sel),f(:,sel)) ;
    %     set(h3,'color','g') ;
    
    %% For each triangle made of the vertices on the box
    for faceIndex = 1:size(faces, 1)
        %% Compute location of the teabox vertices in kth camera coordinate system.
        faceVertices
        
        %% Check if the plane made from the vertices normal makes < 90 degrees, else continue
        
        %% Project each sift point to plane of teabox triangle
        %TriangleRayIntersection('planeType', 'one sided', 'fullReturn', true);
        %% Reject the points which did not intersect with the teabox triangle
    end
end