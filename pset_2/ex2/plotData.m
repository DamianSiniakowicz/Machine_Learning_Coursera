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

accepted_index = find(y==1);
rejected_index = find(y==0);

plot(X(accepted_index,1),X(accepted_index,2), 'k+','LineWidth',2,'MarkerSize',7);
hold on;
plot(X(rejected_index,1),X(rejected_index,2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

% =========================================================================



hold off;

end
