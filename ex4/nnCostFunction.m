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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

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

a1 = [ones(m, 1) X];   %5000x401
z2 = a1 * Theta1';     %5000x25   Theta1 25*401
a2 = sigmoid(z2);      %5000x25
a2 = [ones(m, 1) a2];  %5000x26
z3 = a2 * Theta2';     %5000x10   Theta2 10×26
a3 = sigmoid(z3);      %5000x10
h = a3;                %5000x10

u = eye(num_labels);   %一共有多少个标签
y = u(y,:);            % 5000 * 10 选出每一行的y值作为u的行标，将这行u替换对应行的y，妙啊...
J = 1/m*(sum(sum(-y .* log(h) - (1 - y) .* log(1 - h)))); % 必须要有点 .*
                                                          % sum先把行相加，再把列相加
size(y)

regularization = lambda/(2 * m) * (sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:,2:end) .^ 2)));
J += regularization;

delta3 = a3 - y;                        % 5000 * 10
delta2 = delta3 * Theta2;               % 5000 * 26
delta2 = delta2(:,2:end);               % 5000 * 25
delta2 = delta2 .* sigmoidGradient(z2); % 5000 * 25

Delta1 = zeros(size(Theta1));           % 25 * 401 
Delta2 = zeros(size(Theta2));           % 10 * 26
Delta1 = Delta1 + delta2' * a1;         % 25 * 401 5000×25' * 5000x401 
Delta2 = Delta2 + delta3' * a2;         % 10 * 26  5000×10' * 5000x26
%我们用Delta来表示这个误差矩阵。第L层的第i个激活单元受到第j个参数影响而导致的误差。
Theta1_grad = 1 / m * Delta1 + lambda / m * Theta1;
Theta2_grad = 1 / m * Delta2 + lambda / m * Theta2;
Theta1_grad(:,1) = 1 / m * Delta1(:,1);
Theta2_grad(:,1) = 1 / m * Delta2(:,1);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
