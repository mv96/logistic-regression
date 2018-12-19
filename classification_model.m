function accuracy=classification_model(X_train,y_train,X_test,y_test)
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

%se
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


% Compute accuracy on our training set
check = predict(theta, X_test);
accuracy=mean(double(check == y_test)) * 100;
fprintf('Train Accuracy: %f\n', accuracy);
fprintf('\n');

%calculation part
outliers_number=nnz(check);%non zero elements
actual_outlier_number=nnz(y_test); %actual outliers from the y matrix 
calculations(y_test,check,m,outliers_number,actual_outlier_number);
end