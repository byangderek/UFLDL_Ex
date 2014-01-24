function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
m = size(data,2);
m_inv = 1/m;
for l = 1:(numHidden+1)
    if l==1
        hAct{l} = data;
    else
        hAct{l} = stack{l-1}.W * hAct{l-1} + repmat(stack{l-1}.b,[1,m]);
        if strcmp(ei.activation_fun,'logistic')
            hAct{l} = sigmoid(hAct{l});
        elseif strcmp(ei.activation_fun,'tanh')
            hAct{l} = tanh(hAct{l});
        elseif strcmp(ei.activation_fun,'rlu')
            hAct{l}(hAct{l}<0) = 0;
        end
    end
end
pred_prob = stack{l}.W * hAct{l} + repmat(stack{l}.b,[1,m]);
pred_prob = bsxfun(@minus,pred_prob,max(pred_prob,[],1));
pred_prob = exp(pred_prob);
pred_prob = bsxfun(@rdivide, pred_prob, sum(pred_prob));

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
groundTruth = full(sparse(labels, 1:m, 1));
cost = -mean(sum(groundTruth .* log(pred_prob)));
ceCost = -(groundTruth - pred_prob);
wCost = 0;
for l=1:(numHidden+1)
    wCost = wCost + sum(sum(stack{l}.W .^ 2));
end
wCost = 0.5 * ei.lambda * wCost;
cost = cost + wCost;

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
delta = cell(numHidden+1,1);
for l=numHidden+1:-1:1
    if(l==numHidden+1)
        delta{l} = ceCost;
        gradStack{l}.W = delta{l} * hAct{l}' .* m_inv;
        gradStack{l}.b = mean(delta{l},2);
    else
        delta{l} = stack{l+1}.W' * delta{l+1};
        if strcmp(ei.activation_fun,'logistic')
            delta{l} = delta{l} .* hAct{l+1} .* (1-hAct{l+1});
        elseif strcmp(ei.activation_fun,'tanh')
            delta{l} = delta{l} .* (1-hAct{l+1}.^2);
        elseif strcmp(ei.activation_fun,'rlu')
            delta{l}(hAct{l+1}<0) = 0;
        end
        gradStack{l}.W = delta{l} * hAct{l}' .* m_inv + ei.lambda .* stack{l}.W;
        gradStack{l}.b = mean(delta{l},2);
    end
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end

function sigm = sigmoid(x)
    sigm = 1./(1+exp(-x));
end

