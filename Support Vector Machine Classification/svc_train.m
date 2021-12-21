function      [wh,w0,ap,an,mrg,isp,isn,imp,imn,xip,xin]=svc_train(x,y,c,kernel,kpar,gpar)
%
%
% function    [wh,w0,ap,an,mrg,isp,isn,imp,imn,xip,xin]=svc_train(x,y,c,kernel,kpar,gpar)
%
%  SVM classification by solving dual problem via proximity methods.
%
%  INPUT:     x=[x1,x2,...,xl] inputs points to clssify; xk is nx1 real
%             vector;
%             y=[y1,y2,...,yl] corresponding labels; yk= 1 or -1; 
%             c is the sofmargin parameter; set c=Inf for hard margin
%             classification; 
%             'kernel' is the type of kernel used;
% Options for Kernel function:
%              1) kernel='linear' --> K(u,v)=u*v' 
%              2) kernel='poly'   --> K(u,v)=(u*v'+1)^p1 
%              3) kernel='rbf'    --> K(u,v)=exp(-(u-v)*(u-v)'/(2*p1^2)) 
%              4) kernel='sigmoid'--> K(u,v)=tanh(p1*u*v'/length(u)+p2) 
%              kpar defines corresponding kernel parameters, e.g kpar=[p1, p2]; 
%              gpar=[tol,rho,imax] defines parameters for the proximity 
%              algorithm: tol is zero threshold (1e-10 is default), 
%              rho is ratio presicion (1e-10 is default), imax is max
%              number of iterations (1000 is default)
%
%  OUTPUT:     wh, w0: optimal hyperplane parameters (wh makes sense only
%              for linear kernel) 
%              ap,an: alphas of positive and negative support vectors
%              mrg:   maximum margin achieved
%              isp/isn:   indexes of positive/negative support vectors (0<a<c)
%              (the actual support/margin vectors are obtained as x(:,isp), etc.
%
%              ONLY FOR SOFT-MARGIN CLASSIFICATION (Non-separable case)
%              imp/imn: indexes of positive/negative margin vectors (a=c)
%              xip:     xi's for positive marhin vectors
%              xin:     xi's for negative margin vectors
%              NOTE:  0 <= xi <= 1 indicates margin violation but correct 
%                                  classification
%                     1 < xi       indicates misclassification
%
%              ----------

%   written by A. Sideris, 10/19/04, updated: 3/8/2006,4/24/2006 
%   implements geometrically SMO method
%

tic;
if nargin < 4 || nargin > 6,
    help svmc_train
    return
elseif nargin < 5,
    kpar=[];
    tol=1e-10;
    rho=1e-10;
    imax=1000;
elseif nargin < 6,
    tol=1e-10;
    rho=1e-10;
    imax=1000;
else
    tol=gpar(1);
    rho=gpar(2);
    imax=gpar(3);
end;
tol=1e-8;
rho=1e-1;  % same as SMO epsilon here
rho=rho/2;
imax=100000;
pcount=100;

if strcmp(kernel,'rbf') && length(kpar)~=1,
    disp('Radial Basis Kernel requires one parameter')
    return
end
if strcmp(kernel,'poly') && length(kpar)~=1,
    disp('Polynomial Kernel requires one parameter')
    return
end
if strcmp(kernel,'sigmoid') && length(kpar)~=2,
    disp('Sigmoid Kernel requires two parameters')
    return
end

if c <= 0,
    disp('c must be non-negative; c=Inf, corresponds to HARD-MARGIN case')
    wh=[];ap=[];an=[];w0=[];mrg=[];svp=[];svn=[];...
        xip=[];xin=[];count=0;ierr=5;total_time=0;almeth=[];
    return
end;

if size(x,2) ~= length(y),
    disp(' Number of input points not same with number of labels')
    return
end

% Define polytope constraints;
ipos=find(y==1);
Lp=length(ipos);
ineg=find(y==-1);
Ln=length(ineg);
if isempty(ipos) || isempty(ineg),
    disp('trivial problem---all input points are of the same class')
    return
end
xp=x(:,ipos);
xn=x(:,ineg);
L=Lp+Ln;

% Pick a point in Pos. Polytope
sp=round(rand*(Lp-1)+1);sp=1; % sp are indices of points in positive corral
wp=1;
%cp=1;
% Pick a point in Neg. Polytope
sn=round(rand*(Ln-1)+1);sn=1; % sn are indices of points in negative corral
wn=1;
%cn=1;
% w=1; % only one w, expressing barycentric coordinates wrt corral is necessary!
% n0 (corral size) is the (equal) number of points in cp, cn. Note that
% indices in cp or cn can be repeated as long as no rows of [cp cn] are
% repeated.
n0=1;
np=1; % size of sp
nn=1; % size of sn

% Compute distance between the two points
dist=sqrt(kprod(xp(:,sp),ipos(sp),xp(:,sp),ipos(sp),kernel,kpar,Inf)...
    -2*kprod(xp(:,sp),ipos(sp),xn(:,sn),ineg(sn),kernel,kpar,Inf)...
    +kprod(xn(:,sn),ineg(sn),xn(:,sn),ineg(sn),kernel,kpar,Inf));

lam=min([2/dist^2, c/max(wp), c/max(wn)]);
%i.e since wp=wn=1, lam=min(2/dist^2,c)

%spc and snc are subsets in Lp and Ln with alphas NOT at upper bound (c)
ipc=find(c-wp*lam <1e-10);puv=sp(ipc);spc=(1:Lp);spc(puv)=[];
inc=find(c-wn*lam <1e-10);nuv=sn(inc);snc=(1:Ln);snc(nuv)=[];

% Compute initial Caches for two polytopes and Cross Caches
% Note: Calling kprod with C=Inf does not add 1/C in diagonal elements
% (quadratic case)
kpp=kprod(xp(:,sp),ipos(sp),xp,ipos,kernel,kpar,Inf);
knn=kprod(xn(:,sn),ineg(sn),xn,ineg,kernel,kpar,Inf);
kpn=kprod(xp(:,sp),ipos(sp),xn,ineg,kernel,kpar,Inf);
knp=kprod(xn(:,sn),ineg(sn),xp,ipos,kernel,kpar,Inf);

%isp=1;%isp is positions of sp in kpp(:,1) and kpn(:,1)
%isn=1;%isn is positions of sn in knp(:,1) and knn(:,1)


vpp=kpp;%hpp=0;
vpn=kpn;%hpn=0;
vnp=knp;%hnp=0;
vnn=knn;%hnn=0;
%alpha=1;%hv=1;
%jsp=1;jhp=1;%isp1=1;
%jsn=1;jhn=1;%isn1=1;

% Start outer procedure
%kcmax=500;  % maximum cache size
count=0;
alpha_case=['Gilb';'pMDM';'nMDM';'dMDM'];
almeth=[];%s1=1/4;s2=1;s3=1;s4=1;
while count < imax
    % Check absolute distance criterion
    if dist < tol,
        ierr=1;            % exit with origin being minimum norm point
        break
    end;

    [psv,icp]=min(vpp(spc)-vnp(spc));
    if isempty(icp),
        psv=Inf;
        jsp=[];
    else
        %icp=icp(1); % icp is index of Pos. Gilbert point  (i1)
        icp=spc(icp);
        jsp=find(sp == icp);    % position of icp in sp
    end

    [nsv,icn]=max(vpn(snc)-vnn(snc));
    if isempty(icn),
        nsv=-Inf;
        jsn=[];
    else
        %icn=icn(1); % icn is index of Neg. Gilbert point  (j1)
        icn=snc(icn);
        jsn=find(sn == icn);    % position of icn in sn
    end

    % Update Positive Corral and Cache
    if isempty(jsp) && ~isempty(icp),        % icp is new in sp
        kvp=kprod(xp(:,icp),ipos(icp),xp,ipos,kernel,kpar,Inf);
        kvn=kprod(xp(:,icp),ipos(icp),xn,ineg,kernel,kpar,Inf);
        kpp=[kpp;kvp];
        kpn=[kpn;kvn];
        sp=[sp;icp];np=np+1;
        wp=[wp;0];
        jsp=np;
    end

    % Update Negative Corral and Cache
    if isempty(jsn) && ~isempty(icn) % icn is new in sn
        kvn=kprod(xn(:,icn),ineg(icn),xn,ineg,kernel,kpar,Inf);
        kvp=kprod(xn(:,icn),ineg(icn),xp,ipos,kernel,kpar,Inf);
        knp=[knp;kvp];
        knn=[knn;kvn];
        sn=[sn;icn];nn=nn+1;
        wn=[wn;0];
        jsn=nn;
    end

    % MDM point calculation in each polytope
    [phv,jhp]=max(vpp(sp)-vnp(sp)); %jhp is position of Pos MDM point in sp
    ihp=sp(jhp);     % icp is index of Pos. MDM point (j2)
    [nhv,jhn]=min(vpn(sn)-vnn(sn)); %jhn is position of Neg MDM point in sn
    ihn=sn(jhn);     % icn is index of Neg. MDM point (i2)


    if round(count/pcount) == count/pcount,
        disp(['iter: ',num2str(count),'; pos. corral size: ',num2str(size(sp,1)), ...
            '; neg. corral size: ', num2str(size(sn,1)),'; gap: ',num2str(dist)])
    end

    % Select 2 alphas to mofify in SMO
    dg=lam*[nsv-psv phv-psv nsv-nhv phv-nhv]+[2 0 0 -2];
    [dgmax,id2]=max(dg);
    if -dgmax+2*rho > 0,
        ierr=0;            % exit with minimum point
        break
    end;
    almeth=[almeth id2];


    switch alpha_case(id2,:)

        case 'Gilb',  % Gilbert in Difference polytope (i1,j1)
            % compute norm-squared of contact point (in Diff. polytope)
            % eliminate alphas that will change from upper bound variables
            % (if already there)--- Later if necessary are added back
            ipuv=find(puv == icp);
            if ~isempty(ipuv),
                disp('necessary?'),keyboard
                puv(ipuv)=[];
                spc=[spc icp];
            end
            inuv=find(nuv == icn);
            if ~isempty(inuv),
                disp('necessary?'),keyboard
                nuv(inuv)=[];
                snc=[snc,icn];
            end
            %             spc=union(spc,icp);
            %             snc=union(snc,icn);

            sv=psv-nsv;
            fkfk=kpp(jsp,icp)-kpn(jsp,icn)-knp(jsn,icp)+knn(jsn,icn);
            dk=fkfk-2*sv+dist^2;
            beta=(2/lam-sv)/fkfk; 
            bmin=max([-wp(jsp)  -wn(jsn)]);
            bmax=min([-wp(jsp)+c/lam  -wn(jsn)+c/lam]);
            if beta <= bmin, % at least one Gilbert point weight becomes zero
                beta=bmin;
            elseif beta >= bmax;
                beta=bmax;
                if abs(bmax+wp(jsp)-c/lam) < 1e-10,
                    ipc=find(spc == icp);spc(ipc)=[];
                    puv=[puv icp];
                end
                if abs(bmax+wn(jsn)-c/lam) < 1e-10
                    inc=find(snc == icn);snc(inc)=[];
                    nuv=[nuv icn];
                end
            end
            alpha=beta/(1+beta);
            if alpha == 0, disp('alpha is zero in Gilbert'),count,keyboard,end
            lam=lam/(1-alpha); % or lam=lam*(1+beta)
            wp=(1-alpha)*wp;
            wp(jsp)=wp(jsp)+alpha;
            wn=(1-alpha)*wn;
            wn(jsn)=wn(jsn)+alpha;
            vpp=(1-alpha)*vpp+alpha*kpp(jsp,:);
            vpn=(1-alpha)*vpn+alpha*kpn(jsp,:);
            vnp=(1-alpha)*vnp+alpha*knp(jsn,:);
            vnn=(1-alpha)*vnn+alpha*knn(jsn,:);

        case 'pMDM',     % MDM in Positive polytope (i1,j2)
            % eliminate alphas that will change from upper bound variables
            % (if already there)--- Later if necessary are added back
            ipuv=find(puv == icp);
            if ~isempty(ipuv),
                disp('necessary?'),keyboard
                puv(ipuv)=[];
                spc=[spc icp];
            end
            ipuv=find(puv == ihp);
            if ~isempty(ipuv),
                puv(ipuv)=[];
                spc=[spc ihp];
            end
            dk=kpp(jsp,icp)-2*kpp(jsp,ihp)+kpp(jhp,ihp);
            wkzd=psv-phv;
            alpha=-wkzd/dk;
            almin=max([-wp(jsp) wp(jhp)-c/lam]);
            almax=min([ wp(jhp) c/lam-wp(jsp)]);
            if alpha <= almin,
                alpha=almin;
                if abs(almin-wp(jhp)+c/lam) < 1e-10
                    ipc=find(spc == ihp); spc(ipc)=[];
                    puv=[puv ihp];
                end
            elseif alpha >= almax,
                alpha=almax;
                if abs(almax+wp(jsp)-c/lam) < 1e-10
                    ipc=find(spc == icp); spc(ipc)=[];
                    puv=[puv icp];
                end
            end
            if alpha == 0, disp('alpha is zero in pMDM'),count,keyboard,end
            wp(jhp)=wp(jhp)-alpha;
            wp(jsp)=wp(jsp)+alpha;
            vpp=vpp+alpha*(kpp(jsp,:)-kpp(jhp,:));
            vpn=vpn+alpha*(kpn(jsp,:)-kpn(jhp,:));

        case 'nMDM',     % MDM in Negative polytope (i2,j1)
            % eliminate alphas that will change from upper bound variables
            % (if already there)--- Later if necessary are added back
            inuv=find(nuv == icn);
            if ~isempty(inuv),
                disp('necessary?'),keyboard
                nuv(inuv)=[];
                snc=[snc,icn];
            end
            inuv=find(nuv == ihn);
            if ~isempty(inuv),
                nuv(inuv)=[];
                snc=[snc,ihn];
            end
            dk=knn(jsn,icn)-2*knn(jsn,ihn)+knn(jhn,ihn);
            wkzd=nsv-nhv;
            alpha=wkzd/dk; % note that w_(k+1)=w_k-rho*zd^-
            almin=max([-wn(jsn) wn(jhn)-c/lam]);
            almax=min([ wn(jhn) c/lam-wn(jsn)]);
            if alpha <= almin,
                alpha=almin;
                if abs(almin-wn(jhn)+c/lam) < 1e-10
                    inc=find(snc == ihn); spc(inc)=[];
                    nuv=[nuv ihn];
                end
            elseif alpha >= almax,
                alpha=almax;
                if abs(almax+wn(jsn)-c/lam) < 1e-10
                    inc=find(snc == icn); snc(inc)=[];
                    nuv=[nuv icn];
                end
            end
            if alpha == 0, disp('alpha is zero in nMDM'),count,keyboard,end
            wn(jhn)=wn(jhn)-alpha;
            wn(jsn)=wn(jsn)+alpha;
            vnp=vnp+alpha*(knp(jsn,:)-knp(jhn,:));
            vnn=vnn+alpha*(knn(jsn,:)-knn(jhn,:));

        case 'dMDM'       % optimize along MDM direction (zbar to z)
            % in Difference polytope (i2,j2)
            % compute norm-squared of MDM point (in Diff. polytope)
            % eliminate alphas that will change from upper bound variables
            % (if already there)--- Later if necessary are added back
            ipuv=find(puv == ihp);
            if ~isempty(ipuv),
                puv(ipuv)=[];
                spc=[spc ihp];
            end
            inuv=find(nuv == ihn);
            if ~isempty(inuv),
                nuv(inuv)=[];
                snc=[snc,ihn];
            end
            sv=phv-nhv;
            fkfk=kpp(jhp,ihp)-kpn(jhp,ihn)-knp(jhn,ihp)+knn(jhn,ihn);
            dk=fkfk-2*sv+dist^2;
            beta=(2/lam-sv)/fkfk; %<---correct (SMO) beta
            bmin=max([-wp(jhp)  -wn(jhn)]);
            bmax=min([-wp(jhp)+c/lam  -wn(jhn)+c/lam]);
            if beta <= bmin, % at least one Gilbert point weight becomes zero
                beta=bmin;
            elseif beta >= bmax;
                beta=bmax;
                if abs(bmax+wp(jhp)-c/lam) < 1e-10,
                    ipc=find(spc == ihp);spc(ipc)=[];
                    puv=[puv ihp];
                end
                if abs(bmax+wn(jhn)-c/lam) < 1e-10,
                    inc=find(snc == ihn);snc(inc)=[];
                    nuv=[nuv ihn];
                end
            end
            alpha=beta/(1+beta);
            if alpha == 0, disp('alpha is zero in dMDM'),count,keyboard,end
            lam=lam/(1-alpha); % or lam=lam*(1+beta)
            wp=(1-alpha)*wp;
            wp(jhp)=wp(jhp)+alpha;
            wn=(1-alpha)*wn;
            wn(jhn)=wn(jhn)+alpha;
            vpp=(1-alpha)*vpp+alpha*kpp(jhp,:);
            vpn=(1-alpha)*vpn+alpha*kpn(jhp,:);
            vnp=(1-alpha)*vnp+alpha*knp(jhn,:);
            vnn=(1-alpha)*vnn+alpha*knn(jhn,:);

    end

    % Eliminate zero weights
    izp=find(wp == 0);
    if ~isempty(izp),
        sp(izp)=[]; np=np-length(izp);
        wp(izp)=[]; kpp(izp,:)=[]; kpn(izp,:)=[];
    end
    izn=find(wn == 0);
    if ~isempty(izn),
        sn(izn)=[]; nn=nn-length(izn);
        wn(izn)=[]; knp(izn,:)=[]; knn(izn,:)=[];
    end


    % Compute norm of new point
    new_dist=sqrt(vpp(sp)*wp-2*vpn(sn)*wn+vnn(sn)*wn);
    dist=new_dist;
    count=count+1;

end

disp(['iter: ', num2str(count),'; pos. corral size: ',num2str(size(sp,1)), ...
    '; neg. corral size: ', num2str(size(sn,1)),'; gap: ',num2str(dist)])

% calculate support vectors, alphas, w0 and exit of course input vectors
% are finite-dimensional Note that a new calculation of wp, wn is not
% necessary here


svp=xp(:,sp);
spsp=vpp(sp)*wp;
ap=wp*lam;
imp1=find(abs(ap-c)<=1e-12);
imp=ipos(sp(imp1));
isp=sp;
isp(imp1)=[]; ap(imp1)=[];
isp=ipos(isp);

svn=xn(:,sn);
snsn=vnn(:,sn)*wn;
an=wn*lam;
imn1=find(abs(an-c)<=1e-12);
imn=ineg(sn(imn1));
isn=sn;
isn(imn1)=[]; an(imn1)=[];
isn=ineg(isn);

w0=(snsn-spsp)*lam/2; 
mrg=dist/2;  % maximum margin
wh=svp*[ap;ones(length(imp),1)*c]-svn*[an;ones(length(imn),1)*c]; %maximum margin hyperplane

if c < Inf, %Non-separable case
    xip=ap/c;
    xin=an/c;
else
    xip=[];
    xin=[];
end;


% Set ierr when max iterations are exceeded
if count>= imax,
    ierr=4;               % Exit---Max iterations exceeded
else
    imax=count;
end;

if ierr == 1,
    disp(['Problem is not Linearly Separable, ierr=',num2str(ierr)])
    wh=[];ap=[];an=[];w0=[];mrg=[];svp=[];svn=[];...
        xip=[];xin=[];count=0;ierr=5;total_time=0;almeth=[];
end
total_time=toc
mrg

function k=kprod(u,iu,v,iv,ker,kpar,c)

% NOTE u=[u1,u2,...,un] and v=[v1,v2,...,vm]
% with ui and vj input vectors
% iu (iv) are indexes of u (v) in given data sample
%
% Adding 1/c to diagonal elements of k may not work correctly if neither
% of iu, iv is scalar---standard call case: iu scalar, iv scalar
% or all pos. or neg. points. Modification to fix this---11/16/2005

switch lower(ker)

    case 'linear'
        k = u'*v;
    case 'poly'
        p1=kpar(1);
        k = u'*v;
        k = (k + 1).^p1;
    case 'rbf'
        p1=kpar(1);
        n=size(u,2);
        m=size(v,2);
        k=exp(-((sum(u.^2)')*ones(1,m)-2*u'*v+ones(n,1)*(sum(v.^2)))/2/p1);% changed p1^2 to p1(11/03/2005)
    case 'sigmoid'
        p1=kpar(1);p2=kpar(2);
        k = tanh(p1*u'*v/length(u) + p2);
    otherwise
        disp('Unknown Kernel Function---using Linear kernel')
        k = u'*v;
end

if c < Inf,
    n=size(u,2);
    m=size(v,2);
    for i=1:length(iu),
        ju=find(iv==iu(i));
        k(i,ju)=k(i,ju)+1/c;
    end
end


return



