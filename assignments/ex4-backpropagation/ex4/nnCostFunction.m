function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
% Theta1 is 25 * 401
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

% Theta2 is 10 * 26
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% ----------------------------Part1---------------------------------
% Part1 feedforward with Theta1(25 * 401) and Theta2(10 * 26)
% X is 5000 * 400, y is 5000 * 1
a1 = [ones(m, 1) X];
z2 = a1 * Theta1'; % 5000 * 25
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2]; % 5000 * 26
z3 = a2 * Theta2';
a3 = sigmoid(z3); % 5000 * 10
predictions = a3; % it is 5000 * 10
% we need to recode the labels as vectors containing only values 0 or 1,
% For example, if x(i) is an image of the digit 5, then the corresponding y(i) 
% (that you should use with the cost function) should be a 10-dimensional vector with y5 = 1, 
% and the other elements equal to 0.
transformed_y = zeros(m, num_labels); % 5000 * 10
for i = 1:m,
    transformed_y(i, y(i)) = 1;
end

J = 1 / m * sum(sum(((-transformed_y .* log(predictions)) ... 
        - (ones(m, num_labels) - transformed_y) .* (log(ones(m, num_labels) - predictions))), 2));

% ------------------------------------------------------------------
% ----------------------------Part2---------------------------------
% Implement backpropagation algorithm to generate grad
% 1. implement error delta
error_delta3 = a3 - transformed_y; % matrix 5000 * 10
error_delta2 = error_delta3 * Theta2 .* (a2 .* (1 - a2)); % 5000 * 26
error_delta2 = error_delta2(:, 2:end); % remove the first column, now we get the matrix 5000 * 25
% 2. implement theta grad
Theta1_grad = (1 / m) * (error_delta2' * a1); % 25 * 401
Theta2_grad = (1 / m) * (error_delta3' * a2); % 10 * 26

% ----------------------------Part3---------------------------------
% The cost function for neural networks with regularization
% 1. add regularization to cost fucntion
Theta1_reg = Theta1 .^ 2;
Theta1_reg(:, 1) = zeros(size(Theta1, 1), 1);
Theta2_reg = Theta2 .^ 2;
Theta2_reg(:, 1) = zeros(size(Theta2, 1), 1);
cost_reg = (lambda / (2 * m)) * (sum(Theta1_reg(:)) + sum(Theta2_reg(:)));

J = J + cost_reg;

% 2. add regularization to gradient descent
Theta1_grad_reg = Theta1;
Theta1_grad_reg(:, 1) = zeros(size(Theta1, 1), 1);
Theta2_grad_reg = Theta2;
Theta2_grad_reg(:, 1) = zeros(size(Theta2, 1), 1);

Theta1_grad = Theta1_grad + (lambda / m) * Theta1_grad_reg;
Theta2_grad = Theta2_grad + (lambda / m) * Theta2_grad_reg;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
