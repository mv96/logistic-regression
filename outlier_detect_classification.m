%% Clear and Close Figures
clear ; close all; clc;

%% Load Traning Data and visualize it
fprintf('Loading and Visualizing Data ...\n')
load('arrhythmia.mat');
fprintf('Loading and Visualizing Data ...\n')
m = size(X, 1);
dimensions=size(X, 2);

%Pca implementation
z=pca_x(X);

%shuffle randomly all row
X=[X y];
X= X(randperm(size(X,1)),:);
y= X(:,dimensions+1);
X=X(:,1:dimensions);
data_map(z,y);
%pick 70% of the data as training data and 30% of the data as test data 
[X_train, X_test, y_train, y_test]=divide(X,y);

%plotting the train data 
plotData(X_train,y_train); %plots the training data on the screen
title('Training data(70%)');
xlabel('feature 1');
ylabel('feature 2');
legend('outlier','not an outlier');
plotData(X_test,y_test);   %plots the test data on the screen
xlabel('feature 1');
title('Test data(30%)');
ylabel('feature 2');
legend('outlier','not an outlier');

%%%%%%%%%%%%%%%%%%%Running the classification algorithm%%%%%%%%%%%%%%%%
[m, n] = size(X_train);
[o, p] = size(X_test);
% Add intercept term to x and X_test
X_train = [ones(m, 1) X_train];
X_test  = [ones(o,1)  X_test];
% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X_train, y_train);
fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X_train, y_train)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost)
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Plot Boundary
%plotDecisionBoundary(theta, X_train, y_train);

% Labels and Legend
xlabel('feature 1')
ylabel('feature 2')

% Specified in plot order
legend('non Outlier', 'Outlier')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

prob = sigmoid(X_test* theta);
fprintf(['For testing on test data , we predict a value:  ' ...
         'probability of %f\n'], prob);

% Compute accuracy on our training set
check = predict(theta, X_test);
fprintf('Train Accuracy: %f\n', mean(double(check == y_test)) * 100);
fprintf('\n');

%calculation part
outliers_number=nnz(check);%non zero elements
actual_outlier_number=nnz(y_test); %actual outliers from the y matrix 
calculations(y_test,check,m,outliers_number,actual_outlier_number);
fprintf('\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%cross_validation_approach%%%%%%%%%%%%%%%%%
%pick the cross_validation set split set for 5
splits=5;
[part_1,part_2,part_3,part_4,part_5]=cross_validation(X,y,splits);

%split columns into X and y from parts
dimensions_1 = size(part_1, 2);
dimensions_2 = size(part_2, 2);
dimensions_3 = size(part_3, 2);
dimensions_4 = size(part_4, 2);
dimensions_5 = size(part_5, 2);
part_1x=part_1(:,1:dimensions_1-1);
part_2x=part_2(:,1:dimensions_2-1);
part_3x=part_3(:,1:dimensions_3-1);
part_4x=part_4(:,1:dimensions_4-1);
part_5x=part_5(:,1:dimensions_5-1);
part_1y=part_1(:,dimensions_1);
part_2y=part_2(:,dimensions_2);
part_3y=part_3(:,dimensions_3);
part_4y=part_4(:,dimensions_4);
part_5y=part_5(:,dimensions_5);

%%%%%%test accuracy on all crossvalidation sets%%%%%%%%
accuracy_cross=crossvalidation_test(part_1x,part_2x,part_3x,part_4x,part_5x,
part_1y,part_2y,part_3y,part_4y,part_5y,splits);
accuracy_cross=mean(accuracy_cross)
