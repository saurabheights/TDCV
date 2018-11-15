clear all
clc

% Add auxiliary_code to directory
addpath('auxiliary_code/');

% Load the teabox.ply
[vertex,face] = read_ply('./data/model/teabox.ply');

% CAREFUL
% The vertex index used in ply for a corner, should match the corner index
% given to that corner when the corner pixel location is manually selected.

num_corner = {[1:4,7:8];[1:4,6:8];[1:4,6:7];[1:4,5:7];[1:6];[1:6,8];[1:5,8];[1:5,7:8]};

texture_dir = './data/images/init_texture/';
filePattern = fullfile(texture_dir, '*.JPG');
jpegFiles = dir(filePattern);
numImages = length(jpegFiles);
imgs = zeros(2456, 3680, 3, numImages);
% At most we will need 8 corners for each image.
imgsCornerLoc = cell(numImages);
for k = 1:length(jpegFiles)
    baseFileName = jpegFiles(k).name;
    fullFileName = fullfile(texture_dir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    img = imread(fullFileName);
    imgs(:, :, :, k) = img;
    % get Pixel-Coordinates for corners of teabox in each image
    % Check if we have a corner left to process
    cornerPoint = readPoints(imgs(:, :, :, k), k);
    imgsCornerLoc{k} = cornerPoint;
end

% Save the corner locations in output.txt
save('corner_pixel.mat', imgsCornerLoc);
