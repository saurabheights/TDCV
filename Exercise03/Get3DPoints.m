function [points3d,liesOn3dObject] = Get3DPoints(imagePoints, cameraParams, R, T, vertices, faces)
%% imagePoints should be Nx2
if size(imagePoints,1) == 2 && size(imagePoints, 2) > 2
    imagePoints = imagePoints';
end

[worldOrientation, worldLocation] = extrinsicsToCameraPose(R, T);
points3d = zeros(size(imagePoints,1), 3);
liesOn3dObject = logical(zeros(size(imagePoints,1), 1));
for faceIndex =(1:size(faces,1)) 
    %% Check if faces is seen from the camera
    faceVertices = vertices(faces(faceIndex,:)+1, :);
    faceNormal = cross(faceVertices(1,:) - faceVertices(2,:), ...
    faceVertices(2,:) - faceVertices(3,:));
    faceToCameraVector = worldLocation - mean(faceVertices,1);
    numOfIndices = size(imagePoints, 1);
    if dot(faceNormal, faceToCameraVector) > 0
        orig =repmat(worldLocation, size(imagePoints, 1), 1);
        worldPoints = pointsToWorld(cameraParams, R, T, imagePoints);
        % ToDo: Why the last coordinate is set to zero? It represents z-plane which lie on the bottom of teabox.
        dir = zeros(numOfIndices, 3);
        dir(:, 1:2) = worldPoints;
        dir = dir - orig;
        faceVerticesIndex = faces(faceIndex, :)+1;

        % Find where ray passing through sift points meets the face.
        vert0 = vertices(faceVerticesIndex(1), :);
        vert1 = vertices(faceVerticesIndex(2), :);
        vert2 = vertices(faceVerticesIndex(3), :);
        [INTERSECT, TriangleT, TriangleU, TriangleV, XCOOR] = TriangleRayIntersection(orig, dir, vert0, vert1, vert2, 'planeType', 'one sided', 'fullReturn', true);
        liesOn3dObject = or(liesOn3dObject,INTERSECT);
        points3d(INTERSECT, :) = XCOOR(INTERSECT, :);
    end
end
% End function
end
