function [mean_vec,N_vec,cluster_id] = k_means(x,K,out,miter,tol)
%
% function [mean_vec,N_vec,cid] = k_means(x,K,miter,tol)
%
% Given data in columns of x are grouped into K clusters
% via the k-means algorithm
%
% INPUT:
%       x -- D x N matrix of N, D-dimensional vectors
%       K -- number of desired clusters or a D x K matrix with
%            initial mean vectors
% (optional input arguments)
%       out   --  1  results are plotted in 1d and 2d cases
%                 anything else results in no plots
%       miter --  maximum number of iterations (default:100)
%       tol   --  stopping tolerance for relative change in cluster means
%                 (default:1e-3)
% OUTPUT:
%       mean_vec   -- D x K matrix with columns the cluster mean vectors
%       N_vec      -- 1 x K vector with number of points in each cluster
%       cluster_id -- 1 x N vector with elements in {1,2,...,K} giving the
%                   cluster numbers assigned to corresponding input vectors
%

% Written for MAE 277, Spring 2018, A. Sideris

if nargin < 2
    error('K-means requires at least 2 inputs')
end

[D,N]    = size(x);        % N=number of samples, D=sample dimension
xmin=min(x,[],2);
xmax=max(x,[],2);
if K-round(K)==0 && K > 0    % number of desired clusters is given
    % pick K initial centers uniformly distributed in space of samples
    mean_vec=diag(xmax-xmin)*rand(D,K)+xmin*ones(1,K);
else                       % initial means are given
    mean_vec = K;
    [d,K]    = size(mean_vec);
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

% start k-means iterations
old_mean_vec=zeros(D,K);
iter=1;
tol0 = norm(xmin+xmax)/2;
while norm(mean_vec-old_mean_vec)/tol0 > tol && iter <= miter
    
    old_mean_vec = mean_vec;
    sqdist = inf(K,N);
    % Assign samples to clusters
    for i=1:K
        sqdist(i,:) = sum(bsxfun(@minus,x,mean_vec(:,i)).^2,1);
    end
    [dist2cluster,cluster_id]=min(sqdist,[],1);
    % re-computer cluster means
    for i=1:K
        cid = cluster_id==i;
        if ~any(cid) % there are no points for this cluster;
            % to maintain K clusters, do the following:
            [dummy,ix]=min(sqdist(i,:)); % find sample closest to center
            mean_vec(:,i)=x(:,ix);       % assign center to closest point
            cluster_id(ix)=i;            % re-assign point to cluster
        else
            mean_vec(:,i) = mean(x(:,cid),2);
        end
    end
    
    iter = iter + 1;
    
end
N_vec = zeros(1,K); % number of points in each cluster
for i=1:K
    N_vec(i)=sum(cluster_id==i);
end

if iter == miter
    disp('Maximum number of iterations exceeded')
end

if out == 1
    if D == 2   % plot clusters in 2-D case
        figure(10);clf;
        newplot
        hold on
        
        if K < 8
            cmap = 'brgkmcy'; cmap=cmap';
        else
            k1 = floor(64/K);
            cmap=colormap;
            cmap=cmap((1:k1:k1*K),:);
        end
        
        for i=1:K
            cid = cluster_id==i;
            plot(x(1,cid),x(2,cid),'o','Color', cmap(i,:));
            plot(mean_vec(1,i),mean_vec(2,i),'x','LineWidth',5,...
                'MarkerSize',20,'Color', cmap(i,:));
        end
        title('Results of K-means algorithm')
        hold off
    end
end



