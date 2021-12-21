function y=ffnn2(x,par,w1,b1,w2,b2)
%        y=ffnn2(x,par,w1,b1,w2,b2)
%
%        Evaluates a 2-layer network on given input
%        The neurons on the hidden layer are assumed to have
%        a bipolar sigmoid activation function and the output
%        neurons a linear/bipolar sigmoid/unipolar sigmoid/softmax
%        activation function.
%
%        INPUT:
%           x  - m x p matrix of input vectors.
%           par = [lam, act]
%        where    lam parameter of sigmoid curve.
%                 act=1/2/3/4 linear/bipolar_sigmoid/unipolar_sigmoid/
%                  /softmax activation for output layer; case softmax
%                  reverts to unipolar_sigmoid when output is 1-dimensional
%           w1 - l x m input layer weight matrix.
%           b1 - l x 1 input layer bias vector.
%           w2 - n x l hidden layer weight matrix.
%           b2 - n x 1 hidden layer bias vector.
%         OUTPUT:
%           y  - n x p matrix of output values

%       WRITTEN FOR ME295/MAE277, FALL 1997, UPDATED WINTER 2016 A. SIDERIS
%


% PARAMETERS
lam = par(1);
act = par(2);

if ~any([1 2 3 4]-par(2)==0)
    error(['par(2) should be 1 (linear) or 2 (unipolar_sigmoid) or',...
        ' 3 (bipolar_sigmoid) or 4 (softmax activation) for output activation'])
end

[m, p]=size(x);
if act == 4 && length(b2) == 1
    act=5;  % softmax case with 1 output-->unipolar sigmoid
end

% PRESENTATION
net1=w1*x+b1*ones(1,p);
z=2./(1+exp(-lam*net1))-1;
net2=w2*z+b2*ones(1,p);
switch act
    case 1 % linear
        y=net2;
    case 2 % bipolar sigmoid
        y=2./(1+exp(-lam*net2))-1;
    case 3 % unipolar sigmoid
        y=1./(1+exp(-net2));
    case 4 % softmax, n > 1
        y=exp(net2); y=y/diag(sum(y,1));
    case 5 % softmax, n = 1
        y=1./(1+exp(-net2));
end

