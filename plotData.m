function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
% Find Indices of Positive and Negative Examples
[rows_1 ,coloumns_1] = find(y==1);%store the index of matrix of 1 in pos
[rows_2 ,coloumns_2] = find(y == 0);%store the index of matrix of 0 in ne
% Plot Examples
plot(X(rows_1, 1), X(rows_1, 2), 'k+','LineWidth', 2, ...
'MarkerSize', 7);
plot(X(rows_2, 1), X(rows_2, 2), 'ko', 'MarkerFaceColor', 'y', ...
'MarkerSize', 7);



% =========================================================================



hold off;

end
