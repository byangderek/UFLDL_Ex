function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:(hiddenSize+1)*(numClasses-1)), hiddenSize+1, numClasses-1);
softmaxTheta = softmaxTheta';
softmaxTheta(numClasses,:) = 0;

% Extract out the "stack"
stack = params2stack(theta((hiddenSize+1)*(numClasses-1)+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
M_inv = 1./M;
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

%% forward prop
hAct = cell(numel(stack)+1, 1);

for l = 1:(numel(stack)+1)
    if l==1
        hAct{l} = data;
    else
        hAct{l} = stack{l-1}.w * hAct{l-1} + repmat(stack{l-1}.b,[1,M]);
        hAct{l} = sigmoid(hAct{l});
    end
end

hAct{l}(end+1,:) = 1;
pred_prob = softmaxTheta * hAct{l};
pred_prob = bsxfun(@minus, pred_prob, max(pred_prob,[],1));
pred_prob = exp(pred_prob);
pred_prob = bsxfun(@rdivide, pred_prob, sum(pred_prob));


%% compute cost
eCost = -mean(sum(groundTruth .* log(pred_prob)));
wCost = sum(softmaxTheta(:) .^ 2);
wCost = 0.5 * lambda * wCost;
cost = eCost + wCost;


%% compute gradients using backpropagation
delta = cell(numel(stack)+1,1);

l = 3;
delta{l} = -(groundTruth - pred_prob);
softmaxThetaGrad = delta{l} * hAct{l}' .* M_inv + lambda .* softmaxTheta;
softmaxThetaGrad(end,:) = [];
softmaxThetaGrad = softmaxThetaGrad';

l = 2;
delta{l} = softmaxTheta' * delta{l+1};
delta{l} = delta{l} .* hAct{l+1} .* (1-hAct{l+1});
delta{l}(end,:) = [];
stackgrad{l}.w = delta{l} * hAct{l}' .* M_inv;
stackgrad{l}.b = mean(delta{l},2);

l = 1;
delta{l} = stack{l+1}.w' * delta{l+1};
delta{l} = delta{l} .* hAct{l+1} .* (1-hAct{l+1});
stackgrad{l}.w = delta{l} * hAct{l}' .* M_inv;
stackgrad{l}.b = mean(delta{l},2);


% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
