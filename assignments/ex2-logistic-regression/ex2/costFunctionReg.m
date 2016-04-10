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
predications = sigmoid(X * theta);
cost_items = (y .* log(predications)) + (1 - y) .* log(1 - predications);
reg_items = theta .^ 2;
% don't penalize theta0
sum_reg_items = sum(reg_items) - reg_items(1);
J = (-1 / m) * sum(cost_items) + (lambda / (2 * m)) * sum_reg_items;

partial_derivative_items = (1 / m) * sum((predications - y) .* X)';
% don't penalize theta0, so grad(1) for theta0 is as before
grad(1) = partial_derivative_items(1);
n = size(theta, 1);
for j = 2:n
    grad(j) = partial_derivative_items(j) + (lambda / m) * theta(j);
end


% =============================================================

end
