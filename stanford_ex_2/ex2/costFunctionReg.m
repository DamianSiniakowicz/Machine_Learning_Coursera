function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X * theta;
hypotheses = zeros(m,1);
for one_z = 1:m;
	hypotheses(one_z) = 1 / (1 + exp(-z(one_z)));
	end;
normal_cost = (1/m) * sum((-y .* log(hypotheses) - (1-y).*log(1-hypotheses)));
reg_cost = lambda/(2*m) * sum(theta(2:length(theta)) .^ 2);
J = normal_cost + reg_cost;
pre_grad = (1/m) * ((hypotheses - y)' * X);
pre_grad = pre_grad';
grad_zero = pre_grad(1);
grad_rest = pre_grad(2:length(pre_grad));
grad_rest = grad_rest + (lambda / m)*theta(2:length(theta));
grad = [grad_zero;grad_rest];

% =============================================================

end
