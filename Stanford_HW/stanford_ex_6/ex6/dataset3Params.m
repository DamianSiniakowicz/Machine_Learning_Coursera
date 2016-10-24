function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% my guess is that I'm training or predicting incorrectly

cv_errors = zeros(64,1)

vals = [.01 .03 .1 .3 1 3 10 30] 

% train a model using each C-sigma pair
% we need to pass SVM train the gaussian kernel, but how to pass in sigma?
% check the cross-validation error
for c_value=vals;
	for sigma_value=vals;
		error_index = 8*(find(c_value==vals)-1) + find(sigma_value == vals);
		model = svmTrain(X, y, c_value, @(x1, x2) gaussianKernel(x1, x2, sigma_value));
		predictions = svmPredict(model,Xval);
		cv_error = mean(double(predictions ~= yval));
		cv_errors(error_index) = cv_error;
	end;
end;

C_Index = 0;



min_index = find(min(cv_errors) == cv_errors);
min_index = min_index(1)
%find corresponding C and sig vals
for c_index=1:8;
	if min_index <= c_index * 8;
		C = vals(c_index);
		C_Index = c_index;
		break;
	end;
end;

C_Index

sig_index = min_index - 8 * (C_Index - 1);

sig_index

sigma = vals(sig_index);
% =========================================================================

end
