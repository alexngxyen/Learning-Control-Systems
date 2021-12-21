function [fw1,fb1,fw2,fb2,temin,vemin,iter,ter,ver]=bp2ma(x,d,par,val_per,opt,lr_adapt,w1,b1,w2,b2)
%        [fw1,fb1,fw2,fb2,temin,vemin,iter,ter,ver]=bp2ma(x,d,par,val_per,opt,lr_adapt,w1,b1,w2,b2)
%
%        Trains a 2-layer network with backpropagation
%        The neurons on the hidden layer are assumed to have
%        a bipolar sigmoid activation function and the output
%        neurons a linear, bipolar sigmoid, unipolar sigmoid, or a softmax
%        activation function. In the first three cases a quadratic loss
%        function  while in the last case the cross-entropy loss function
%        is minimized. Also case softmax reverts to unipolar_sigmoid when
%        output is 1-dimensional (a 2-class classification problem)
%
%        INPUT:
%           x  - m x p matrix of input vectors.
%           d  - n x p matrix of target vectors.
%           par - vector of training parameters:
%                [l, max_iter, max_error]
%        where
%           l          number of neurons on hidden layer.
%           max_iter   maximum number of batch iterations.
%           max_error  maximum permissible square error per pattern.
%
%        Optional: automatically initialized if not given or set to [].
%           val_per -  in [0,0.5] fraction of data reserved for validation;
%                      set to 0 (default) for no validation
%           opt -training options: opt=[act,lam,learn_rate,mbatch,out]
%                (default: opt=[1 1 1e-3 inf 0 0 1]).
%             act=1/2/3/4 linear/bipolar_sigmoid/unipolar_sigmoid/
%                  /softmax activation for output layer; case softmax
%                  reverts to unipolar_sigmoid when n=1
%             lam - parameter of sigmoid curve
%             learn_rate - (initial) learning rate.
%             mbatch - in (0,1]: mini-batch size as a fraction of training data
%             rho1/rho2 -  regularization constants for minimizing 
%                          "error function"+rho1/2*||W1||^2+rho2/2*||W2||^2
%                           for weights W1/W2 of hidden/output layers
%             out= K: output messages during training are displayed
%                     every K iterations; set K=0 for no output messages.
%             lr_adapt - options for learning rate; acceptable values:
%                        'const','ada','mom','adamon', 'adagrad',
%                        'RMSProp'(default),'adam'}
%
%           w1 - l x m layer-1 weight matrix.
%           b1 - l x 1 layer-1 bias vector.
%           w2 - n x l layer-2 weight matrix.
%           b2 - n x 1 layer-2 bias vector.
%         OUTPUT:
%           fw1,fb1 - final weights & biases for input layer.
%           fw2,fb2 - final weights & biases for hidden layer.
%           temin   - minimum training error achieved (results in final
%                       NN weights if validation is not used)
%           vemin   - minimum validation error achieved (results in final
%                       NN weights if validation is used)
%           iter  - the actual number of batch iterations trained.
%           ter - vector of training error per sample at each iteration.
%           ver - vector of validation error per sampleat each iteration.

%
%         WRITTEN FOR ME295/277, FALL 1997, updated WINTER 2016, SPRING 2018
%         A. SIDERIS

if nargin < 3
    error(' At least 3 input arguments are required')
end
% TRAINING PARAMETERS
l   = round(par(1));
mi  = par(2);
me  = par(3);

[m, p]=size(x);
[n, p1]=size(d);

if p~=p1
    error('row dimensions of x and d must equal to the number of training patterns')
end

if nargin < 4 || isempty(val_per),
    val_per = 0; % no validation will be performed
elseif val_per*(1-val_per) < 0
    error('validation fraction should be in [0, 0.5]')
end

if nargin < 5 || isempty(opt),
    act=1;         % linear activation at output layer
    lam=1;         % sigmoid parameter
    eta=1e-3;      % initial learn_rate
    mb=1;          % mini-batch size as a fraction of training data
    rho1=0;rho2=0; % no regularization
    out=10^max(0,log10(mi)-2);      % messages during training
elseif  ~any([1 2 3 4]-opt(1)==0)
    error(['opt(1) should be 1 (linear) or 2 (unipolar_sigmoid) or',...
        ' 3 (bipolar_sigmoid) or 4 (softmax activation) for output activation'])
elseif opt(2) <= 0,
    error('opt(2)--sigmoid lambda should be positive')
elseif opt(3) < 0
    error('opt(3)--learning rate should be positive')
elseif opt(4)<=0 || opt(4)>1
    error('opt(4)-mini-batch size as a fraction of training data should be in (0,1]')
elseif opt(5)<0 || opt(6)<0
    error('reguralization parameters (opt(5)/opt(6)) must be non-negative')
elseif opt(7)<0,
    error('opt(7) should be K>1 (messages) or 0 (no messages)')
else
    act=opt(1);  % activation for output neurons
    lam=opt(2);
    eta=opt(3);
    mb=opt(4);
    rho1=opt(5);
    rho2=opt(6);
    out=opt(7);
end
if act == 4 && n == 1
    act=5;  % softmax case with 1 output-->unipolar sigmoid
end

if nargin < 6 || isempty(lr_adapt),
    lr_adapt ='RMSProp';
end

% WEIGHT INITIALIZATION
if nargin < 7 % no weights are given
    sx=sqrt(max(sum(x.^2,1)+1));
    w1=randn(l,m)/sx; b1=randn(l,1)/sx;
    w2=randn(n,l)/sqrt(l+1); b2=randn(n,1)/sqrt(l+1);
else
    if nargin~=10
        error('not enough initial weights are given')
    elseif sum(size(w1)-[l,m])+sum(size(b1)-[l 1])+...
            sum(size(w2)-[n,l])+sum(size(b2)-[n 1])~=0
        error('dimensions of initial weights are not consistent')
    end
end

% create training and validation sets
id = randperm(p);
x = x(:,id);
d = d(:,id);
id=(1:1:p);
lv = round(p*val_per); % size of validation set
lt = p-lv;             % size of training set
iv = id(1:lv);         % pattern indices reserved for validation
it = id(lv+1:p);       % pattern indices reserved for training
xt=x(:,it); dt=d(:,it);           % training patterns
xv=x(:,iv); dv=d(:,iv);           % validation patterns

% FIRST PRESENTATION
net1=w1*x+b1*ones(1,p);
z=2./(1+exp(-lam*net1))-1;
net2=w2*z+b2*ones(1,p);
switch act
    case 1 % linear
        y=net2;
        e=d-y;
        te=sum(sum(e(:,it).*e(:,it)))/lt;
        ve=sum(sum(e(:,iv).*e(:,iv)))/lv;
    case 2 % bipolar sigmoid
        y=2./(1+exp(-lam*net2))-1;
        e=d-y;
        te=sum(sum(e(:,it).*e(:,it)))/lt;
        ve=sum(sum(e(:,iv).*e(:,iv)))/lv;
    case 3 % unipolar sigmoid
        y=1./(1+exp(-lam*net2));
        e=d-y;
        te=sum(sum(e(:,it).*e(:,it)))/lt;
        ve=sum(sum(e(:,iv).*e(:,iv)))/lv;
    case 4 % softmax, n > 1
        y=exp(net2); y=y/diag(sum(y,1));
        e=d-y;
        te=-sum(sum(d(:,it).*log(y(:,it)+eps)))/lt;
        ve=-sum(sum(d(:,iv).*log(y(:,iv)+eps)))/lv;
    case 5 % softmax, n =1
        y=1./(1+exp(-net2));
        e=d-y;
        te=-sum(sum(d(:,it).*log(y(:,it)+eps)+(1-d(:,it)).*log(1-y(:,it)+eps)))/lt;
        ve=-sum(sum(d(:,iv).*log(y(:,iv)+eps)+(1-d(:,iv)).*log(1-y(:,iv)+eps)))/lv;
end
temin=te;
vemin=inf;
fw1=w1; fb1=b1;
fw2=w2; fb2=b2;
ter=te; ver=ve;
er1=te;

tw1=w1; tb1=b1;
tw2=w2; tb2=b2;
dw10=zeros(size(w1));  db10=zeros(size(b1));
dw20=zeros(size(w2));  db20=zeros(size(b2));
dv10=zeros(size(w1));  dc10=zeros(size(b1));
dv20=zeros(size(w2));  dc20=zeros(size(b2));

lb = ceil(lt*mb);              % actual mini-batch size
p1 = (1:lb:lt);                % start indices of mini-batches
p2 = (lb:lb:lt);               % end indices of mini-batches
if length(p1) > length(p2)
    p2(end) = lt;
    p1 = p1(1:end-1);
end
nb = length(p2);               % number of mini-batches
if length(p1)~=length(p2) || p2(end)~=lt
    disp('mini-batch segmentation failed!')
    keyboard
end

iter=0;
beta1=0.9;beta1t=1;
beta2=0.999;beta2t=1;
xt_sav=xt;
dt_sav=dt;
while iter < mi && temin > me && (lv == 0 || ve <= 1.25*vemin) % && abs(dtse0-dtse) > sqrt(eps)

    id = randperm(lt);
    xt=xt(:,id);
    dt=dt(:,id);

    iter = iter + 1;
    te=0; % initialize total square error over mini-batches
    for i=1:nb % cycle through mini-batches

        i1=(p1(i):p2(i));
        l1=length(i1);

        % PRESENTATION PHASE
        net1=tw1*xt(:,i1)+tb1*ones(1,l1);
        z=2./(1+exp(-lam*net1))-1;
        net2=tw2*z+tb2*ones(1,l1);
        switch act
            case 1 % linear
                y=net2;
                e=dt(:,i1)-y;
                %                     te=sum(sum(e.*e))/l1;
                tei=sum(sum(e.*e));

            case 2 % bipolar sigmoid
                y=2./(1+exp(-lam*net2))-1;
                e=dt(:,i1)-y;
                %                     te=sum(sum(e.*e))/l1;
                tei=sum(sum(e.*e));
            case 3 % unipolar sigmoid
                y=1./(1+exp(-lam*net2));
                e=dt(:,i1)-y;
                %                     te=sum(sum(e.*e))/l1;
                tei=sum(sum(e.*e));

            case 4 % softmax, n > 1
                y=exp(net2); y=y/diag(sum(y,1));
                e=dt(:,i1)-y;
                %                     te=-sum(sum(dt(:,i1).*log(y))/l1);
                tei=-sum(sum(dt(:,i1).*log(y+eps)));
            case 5 % softmax, n =1
                y=1./(1+exp(-net2));
                e=dt(:,i1)-y;
                %                     te=-sum(sum(dt(:,i1).*log(y)+(1-dt(:,i1)).*log(1-y)))/l1;
                tei=-sum(sum(dt(:,i1).*log(y+eps)+(1-dt(:,i1)).*log(1-y+eps)));
        end
        er0=er1; er1=tei/l1;

        % BACKPROPAGATION PHASE
        switch act
            case 1 % linear
                d2=e;
            case 2 % bipolar sigmoid
                d2=e.*(1-y.^2)*(lam/2);
            case 3 % unipolar sigmoid
                d2=e.*y.*(1-y)*lam;
            case 4 % softmax, n > 1
                d2=e;
            case 5 % softmax, n =1
                d2=e;
        end
        d1=(tw2'*d2).*(1-z.^2)*(lam/2);

        % LEARNING PHASE
        switch lr_adapt

            case 'const'
                tw1=tw1+eta*(d1*xt(:,i1)'-rho1*tw1);
                tb1=tb1+eta*(d1*ones(l1,1)-rho1*tb1);
                tw2=tw2+eta*(d2*z'-rho2*tw2);
                tb2=tb2+eta*(d2*ones(l1,1)-rho2*tb2);

            case 'ada'
                if er1 <= er0
                    tw1=tw1+eta*(d1*xt(:,i1)'-rho1*tw1);
                    tb1=tb1+eta*(d1*ones(l1,1)-rho1*tb1);
                    tw2=tw2+eta*(d2*z'-rho2*tw2);
                    tb2=tb2+eta*(d2*ones(l1,1)-rho2*tb2);
                    eta=min(eta*1.05,50);
                else
                    eta=max(eta*0.7,1e-6);
                end

            case 'mom'
                dw10=beta1*dw10+eta*(d1*xt(:,i1)'-rho1*tw1);
                db10=beta1*db10+eta*(d1*ones(l1,1)-rho1*tb1);
                dw20=beta1*dw20+eta*(d2*z'-rho2*tw2);
                db20=beta1*db20+eta*(d2*ones(l1,1)-rho2*tb2);

                tw1=tw1+dw10;
                tb1=tb1+db10;
                tw2=tw2+dw20;
                tb2=tb2+db20;

            case 'adamom'
                if er1 <= er0
                    dw10=beta1*dw10+eta*(d1*xt(:,i1)'-rho1*tw1);
                    db10=beta1*db10+eta*(d1*ones(l1,1)-rho1*tb1);
                    dw20=beta1*dw20+eta*(d2*z'-rho2*tw2);
                    db20=beta1*db20+eta*(d2*ones(l1,1)-rho2*tb2);

                    tw1=tw1+dw10;
                    tb1=tb1+db10;
                    tw2=tw2+dw20;
                    tb2=tb2+db20;
                    eta=min(eta*1.05,50);
                else
                    eta=max(eta*0.7,1e-6);
                end

            case 'adagrad'
                dv10=dv10+(d1*xt(:,i1)'-rho1*tw1).^2;
                dc10=dc10+(d1*ones(l1,1)-rho1*tb1).^2;
                dv20=dv20+(d2*z'-rho2*tw2).^2;
                dc20=dc20+(d2*ones(l1,1)-rho2*tb2).^2;

                tw1=tw1+eta*(d1*xt(:,i1)'-rho1*tw1)./(sqrt(dv10)+1e-8);
                tb1=tb1+eta*(d1*ones(l1,1)-rho1*tb1)./(sqrt(dc10)+1e-8);
                tw2=tw2+eta*(d2*z'-rho2*tw2)./(sqrt(dv20)+1e-8);
                tb2=tb2+eta*(d2*ones(l1,1)-rho2*tb2)./(sqrt(dc20)+1e-8);

            case 'RMSProp'
                dv10=beta2*dv10+(1-beta2)*(d1*xt(:,i1)'-rho1*tw1).^2;
                dc10=beta2*dc10+(1-beta2)*(d1*ones(l1,1)-rho1*tb1).^2;
                dv20=beta2*dv20+(1-beta2)*(d2*z'-rho2*tw2).^2;
                dc20=beta2*dc20+(1-beta2)*(d2*ones(l1,1)-rho2*tb2).^2;

                tw1=tw1+eta*(d1*xt(:,i1)'-rho1*tw1)./(sqrt(dv10)+1e-8);
                tb1=tb1+eta*(d1*ones(l1,1)-rho1*tb1)./(sqrt(dc10)+1e-8);
                tw2=tw2+eta*(d2*z'-rho2*tw2)./(sqrt(dv20)+1e-8);
                tb2=tb2+eta*(d2*ones(l1,1)-rho2*tb2)./(sqrt(dc20)+1e-8);

            case 'adam'
                beta1t=beta1t*beta1;
                beta2t=beta2t*beta2;

                dw10=(beta1*dw10+(1-beta1)*d1*xt(:,i1)'-rho1*tw1);
                db10=(beta1*db10+(1-beta1)*d1*ones(l1,1)-rho1*tb1);
                dw20=(beta1*dw20+(1-beta1)*d2*z'-rho2*tw2)/(1-beta1t);
                db20=(beta1*db20+(1-beta1)*d2*ones(l1,1)-rho2*tb2);

                dv10=(beta2*dv10+(1-beta2)*(d1*xt(:,i1)'-rho1*tw1).^2);
                dc10=(beta2*dc10+(1-beta2)*(d1*ones(l1,1)-rho1*tb1).^2);
                dv20=(beta2*dv20+(1-beta2)*(d2*z'-rho2*tw2).^2)/(1-beta2t);
                dc20=(beta2*dc20+(1-beta2)*(d2*ones(l1,1)-rho2*tb2).^2);

                tw1=tw1+eta*(dw10/(1-beta1t))./(sqrt(dv10/(1-beta2t))+1e-8);
                tb1=tb1+eta*(db10/(1-beta1t))./(sqrt(dc10/(1-beta2t))+1e-8);
                tw2=tw2+eta*(dw20/(1-beta1t))./(sqrt(dv20/(1-beta2t))+1e-8);
                tb2=tb2+eta*(db20/(1-beta1t))./(sqrt(dc20/(1-beta2t))+1e-8);

            otherwise
                error('unknown learning rate adaptation rule')

        end
        te = te + tei;

    end
    te = te /lt; % average square error over a round of mini-batches

    if nb ~= 1
        xt=xt_sav;
        dt=dt_sav;
    end

    % VALIDATION ERROR
    net1=tw1*xv+tb1*ones(1,lv);
    z=2./(1+exp(-lam*net1))-1;
    net2=tw2*z+tb2*ones(1,lv);

    switch act
        case 1 % linear
            y=net2;
            e=dv-y;
            ve=sum(sum(e.*e))/lv;
        case 2 % bipolar sigmoid
            y=2./(1+exp(-lam*net2))-1;
            e=dv-y;
            ve=sum(sum(e.*e))/lv;
        case 3 % unipolar sigmoid
            y=1./(1+exp(-lam*net2));
            e=dv-y;
            ve=sum(sum(e.*e))/lv;
        case 4 % softmax, n > 1
            y=exp(net2); y=y/diag(sum(y,1));
            e=dv-y;
            ve=-sum(sum(dv.*log(y)+eps))/lv;
        case 5 % softmax, n =1
            y=1./(1+exp(-net2));
            e=dv-y;
            ve=-sum(sum(dv.*log(y+eps)+(1-dv).*log(1-y+eps)))/lv;

    end

    %   save best weights
    if ve < vemin,
        vemin=ve;
        fw1=tw1; fb1=tb1;
        fw2=tw2; fb2=tb2;
    end
    if te < temin,
        temin=te;
        if val_per==0
            fw1=tw1; fb1=tb1;
            fw2=tw2; fb2=tb2;
        end
    end
    ter=[ter te];
    ver=[ver ve];

    % DISPLAY RESULTS
    if mod(iter,out)+1 == 1,
        fprintf('Iteration %.0f: Train error %g, Validation error %g.\n',iter,te,ve)
    end;

end;

if out ~= 0,
    figure(1); cla reset
    if val_per == 0
        fprintf('Fig. 1: Total training error during training\n')
        i1=[0:1:length(ter)-1];
        plot(i1, ter)
        title('Total training error during training');
    else
        fprintf('Fig. 1: Total training/validation errors during training\n')
        i1=[0:1:length(ter)-1];
        plot(i1, [ter;ver])
        title('Total training/validation errors during training');
        legend({'Training Error','Validation Error'})
    end
    xlabel('Iterations');
    ylabel('Error');

end
