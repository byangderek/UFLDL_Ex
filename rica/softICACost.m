%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%
cost1 = sqrt((W*x).^2 + params.epsilon);
cost1 = params.lambda * sum(cost1(:));
delta = W'*W*x - x;
cost2 = delta.^2;
cost2 = 0.5 * sum(cost2(:));
cost = (cost1 + cost2) / size(x,2);

delta1 = (W*x)./sqrt((W*x).^2+params.epsilon);
Wgrad1 = params.lambda.*delta1*x';
Wgrad2 = (W*delta)*x' + (W*x)*delta';
Wgrad = (Wgrad1+Wgrad2) ./ size(x,2);

% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);