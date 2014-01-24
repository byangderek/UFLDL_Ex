function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:(hiddenSize+1)*(numClasses-1)), hiddenSize+1, numClasses-1);
softmaxTheta = softmaxTheta';
softmaxTheta(end+1,:) = 0;

% Extract out the "stack"
stack = params2stack(theta((hiddenSize+1)*(numClasses-1)+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

M = size(data,2);

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

[~,pred] = max(pred_prob);
pred = pred';

% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
