function [mu,var,wgt,llh] = EM(x,K,out,miter,tol)
%
% function [mean_vec,N_vec,cid] = k_means(x,K,miter,tol)
%
% Given data in columns of x are modeled as a Gaussian mixture with K
% components via the EM (Expectation-Maximization) algorithm.
%
% INPUT:
%       x -- D x N matrix of N, D-dimensional vectors
%       K -- number of desired components (clusters) or a D x K matrix with
%            initial mean vectors
% (optional input arguments)
%       out   --  1  results are plotted in 1d and 2d cases
%                 anything else results in no plots
%       miter --  maximum number of iterations (default:100)
%       tol   --  stopping tolerance for relative change in  means or
%                 log-likelihood (default:1e-3)
% OUTPUT:
%       mu   -- D x K matrix with columns the mixture mean vectors
%       var  -- D x D x K matrix with pages the mixture covariance matrices
%       wgt  -- 1 x K vector of the mizture component probabilities
%               Then:  x ~ sum_{1:K} {wgt(k)*Normal(x|mu(k),var(:,:,k)}
%       llh  -- log-likelihood during iterations (it should be always
%               non-decreasing)
%

% Written for MAE 277, Spring 2018, A. Sideris

[D,N]    = size(x);        % N=number of samples, D=sample dimension
xmin=min(x,[],2);
xmax=max(x,[],2);
if K-round(K)==0 && K > 0 % number of desired clusters is given
    % pick K initial centers uniformly distributed in space of samples
    mu_vec=diag(xmax-xmin)*rand(D,K)+xmin*ones(1,K);
else                       % initial means are given
    mu_vec = K;
    [d,K]  = size(mu_vec);
    if d ~= D
        error('initial mean vectors and input vectors have different dimensions')
    end
end
if nargin < 3
    out=0;
    miter = 100;
    tol = 1e-3;
elseif nargin < 4
    miter = 100;
    tol = 1e-3;
elseif nargin < 5
    tol =  1e-3;
end


% initialize other variables
Sigma_mat = zeros(D,D,K); % cluster covariances
R         = zeros(K,N);   % responsibilities
N_vec     = zeros(1,K);   % effective number of cluster elements
Pi        = zeros(1,K);   % cluster probabilities (weights)
% % Assign randomly samples to clusters
% R=rand(K,N);
% R_sum     = sum(R)+eps;
% R         = bsxfun(@ldivide,R_sum,R);  % normalize rows of R

% Hard-Assign samples to clusters based on distance from centers
for i=1:N
    temp = sum(bsxfun(@minus,mu_vec,x(:,i)).^2,1);
    temp = cumsum(temp/sum(temp));
%     R(:,i) = R(:,i)==min(R(:,i));
    k = find(temp > rand, 1);
    R(k,i) = 1;
end

% Estimate initial covariance matrices
% Regularize Sigma to avoid singularities;  see Murphy book eq. (11.48)
% Sigma_0 = diag(diag(cov(x')))/K^(1/D); % see Murphy book eq. (11.48)
Sigma_0 = diag(sum((bsxfun(@minus, x, mean(x,2))).^2,2))/(K^(1/D)*N);
for j=1:K
    n_j              = sum(R(j,:));
    x_bar            = bsxfun(@minus,x,mu_vec(:,j));
    Sigma_temp       = x_bar*bsxfun(@times,R(j,:),x_bar)';
    Sigma_mat(:,:,j) = (Sigma_temp+Sigma_0)/(n_j+2*D+4);
end

% start EM iterations
old_mu_vec=zeros(D,K);
iter=1;
llh=inf;llh0=1;
tol0=norm(xmin+xmax)/2;
while (norm(mu_vec-old_mu_vec)/tol0 > tol || ...
        abs((llh(iter)-llh0)/llh0) > tol) && iter <= miter
    
    old_mu_vec = mu_vec;
    llh0       = llh(iter);
    % M-step: update cluster means and covariances,
    for j = 1:K
        n_j              = sum(R(j,:));
        mu_vec(:,j)      = 1/n_j*sum(bsxfun(@times,R(j,:),x),2);
        x_bar            = bsxfun(@minus,x,mu_vec(:,j));
        Sigma_temp       = x_bar*bsxfun(@times,R(j,:),x_bar)';
        Sigma_mat(:,:,j) = (Sigma_temp+Sigma_0)/(n_j+2*D+4);
        if any(isnan(Sigma_mat(:,:,j)));keyboard;end
        Pi(j)            = n_j/N;
        N_vec(j)         = n_j;
    end
    
    
    % E-step
    for j =1:K
        R(j,:) = mvnpdf(x',mu_vec(:,j)',Sigma_mat(:,:,j))'*Pi(j);
    end
    R_sum     = sum(R)+eps;
    R         = bsxfun(@ldivide,R_sum,R);  % normalize rows of R
    
    % Log Likelihood computation
    llh       = [llh sum(log(R_sum))];
    
    iter      = iter + 1;
    
end


% outputs are:
mu      =  mu_vec;
var     = Sigma_mat;
wgt     = Pi;
llh     = llh(2:end);

if out == 1
    figure(20);clf
    if D ==1
        x_axis     = linspace(xmin,xmax,400)';
        y_data     = zeros(400,1);
        for j=1:K
            y_data   = y_data + Pi(j)* normpdf(x_axis,mu_vec(j),Sigma_mat(:,:,j));
        end
        h_EM       = plot(x_axis,y_data);
        axis([xmin xmax min(y_data) max(y_data)])
        set(h_EM, 'Color', 'red','Linewidth',3);
        hold on
        [f1,x1]=hist(x,100);
        bar(x1,f1/trapz(x1,f1))
    elseif D ==2
        newplot;clf
        hold on
        if K < 8
            cmap = 'brgkmcy'; cmap=cmap';
        else
            k1 = floor(64/K);
            cmap=colormap;
            cmap=cmap((1:k1:k1*K),:);
        end
        
        [Pi_max,cluster_id]=max(R,[],1);
        for i=1:K
            cid = cluster_id==i;
            plot(x(1,cid),x(2,cid),'o','Color', cmap(i,:));
            plot(mu_vec(1,i),mu_vec(2,i),'x','LineWidth',5,...
                'MarkerSize',20,'Color', cmap(i,:));
            out = draw_ellipse(mu_vec(:,i),Sigma_mat(:,:,i));
            plot(out(:,1),out(:,2),'Color', cmap(i,:),'Linewidth',3);
        end
        title('Results of EM algorithm')
        hold off
        
    end
end

function output = draw_ellipse(mean_vec, covariance)

% this is a modified verstion of code that I got from:
% http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/

theta_grid       = linspace(0,2*pi,20);

% Calculate the eigenvectors and eigenvalues
[eigenvec, eigenval ] = eig(covariance);

% Get the index of the largest eigenvector
[largest_eigenvec_ind_c, r] = find(eigenval == max(max(eigenval)));
largest_eigenvec = eigenvec(:, largest_eigenvec_ind_c);

% Get the largest eigenvalue
largest_eigenval = max(max(eigenval));

% Get the smallest eigenvector and eigenvalue
if(largest_eigenvec_ind_c == 1)
    smallest_eigenval = max(eigenval(:,2));
    smallest_eigenvec = eigenvec(:,2);
else
    smallest_eigenval = max(eigenval(:,1));
    smallest_eigenvec = eigenvec(1,:);
end

% Calculate the angle between the x-axis and the largest eigenvector
angle = atan2(largest_eigenvec(2), largest_eigenvec(1));

% This angle is between -pi and pi.
% Let's shift it such that the angle is between 0 and 2pi
if(angle < 0)
    angle = angle + 2*pi;
end

% Get the coordinates of the data mean
X0 = mean_vec(1);
Y0 = mean_vec(2);

% Get the 95% confidence interval error ellipse
chisquare_val = 2.4477;
% theta_grid = linspace(0,2*pi,20);
phi = angle;
a = chisquare_val*sqrt(largest_eigenval);
b = chisquare_val*sqrt(smallest_eigenval);

% the ellipse in x and y coordinates
ellipse_x_r  = a*cos( theta_grid );
ellipse_y_r  = b*sin( theta_grid );

%Define a rotation matrix
R = [ cos(phi) sin(phi); -sin(phi) cos(phi) ];

%let's rotate the ellipse to some angle phi
r_ellipse = [ellipse_x_r;ellipse_y_r]' * R;

% Draw the ellipse
output = [r_ellipse(:,1) + X0, r_ellipse(:,2) + Y0];
%plot(r_ellipse(:,1) + X0,r_ellipse(:,2) + Y0,'-')





