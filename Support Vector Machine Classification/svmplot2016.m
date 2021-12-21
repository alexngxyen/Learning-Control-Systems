% Sample points are denoted as (x,y) 
% Specify next training parameters C, kernel, gpar
% C=.5; % use the value of C obtained from cross-validation here
% kernel='rbf';
% kpar=1;

C = C_opt;

figure; clf;
hold on

% it is assumed that(x, y) is the training data 
% we distinguish positive and negative patterns
xp=x(:,find(y==1));
xn=x(:,find(y==-1));

% to train a SUPPORT VECTOR CLASSIFIER with the command:
[wh,w0,ap,an,mrg,isp,isn,imp,imn,xip,xin]=svc_train(x,y,C,kernel,kpar);
%(read the "help listing" of svc_train for the meaning of the variables)
% 
% we can then obtain:
svp=x(:,isp); % positive support vectors
svn=x(:,isn); % negative support vectors
mvp=x(:,imp); % positive margin vectors
mvn=x(:,imn); % negative margin vector
% to be used for plotting the decision boundaries
% a grid of points spanning the space of the data is obtained next
rscl = 1.01; gtol=0.1;
xa_min = min(x(1,:))/rscl;
xa_max = max(x(1,:))*rscl;
ya_min = min(x(2,:))/rscl;
ya_max = max(x(2,:))*rscl;
nxa    = ceil((xa_max-xa_min)/gtol);
nya    = ceil((ya_max-ya_min)/gtol);
xa_pts = linspace(xa_min,xa_max,nxa);
ya_pts = linspace(ya_min,ya_max,nya);
[X,Y] = meshgrid (xa_pts,ya_pts);
xt=[X(:)';Y(:)'];

% SVC classification of xt points
yt=svc(xt,svp,ap,svn,an,mvp,mvn,C,w0,kernel,kpar);
% plot decision boundaries of SVC classifier
Z=reshape(yt,nya,nxa);
[cc h] = contour(X,Y,Z,[0 0],'k-');
set(h,'linewidth',1.5)

% plot margin lines
[cc1,h1] =  contour(X,Y,Z,[-2*mrg,-2*mrg],'--b');
set(h1,'linewidth',1.5)
[cc2,h2] =  contour(X,Y,Z,[2*mrg, 2*mrg],'--r');
set(h2,'linewidth',1.5)
% mark support vectors (works for hard margin linear case)
plot(svp(1,:),svp(2,:),'sr');
plot(svn(1,:),svn(2,:),'sb');

% mark positive and negative regions
xtp=xt(:,yt > 0);
xtn=xt(:,yt < 0);
plot(xtp(1,:),xtp(2,:),'.','Color',[255,182,193]/256)
plot(xtn(1,:),xtn(2,:),'.','Color',[224,255,255]/256)
% plot training data
plot(xp(1,:),xp(2,:),'xr')
plot(xn(1,:),xn(2,:),'ob')

title({'\bf Support Vector Machine Classification of Banana Data';...
    ['  C=',num2str(C),',', strcat('  Kernel: ',kernel),',',...
    '  kernel parameters:  ',num2str(kpar)]});

hold off
shg

