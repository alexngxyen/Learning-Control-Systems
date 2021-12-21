function    y=svc(x,svp,ap,svn,an,mvp,mvn,C,w0,kernel,kpar)

%       y=svc(xt,svp,ap,svn,an,mvp,mvn,C,b0,kernel,kpar)
%
%       SVC  computes the output of a trained SUPPORT VECTOR CLASSIFIER 
%            corresponding to the input samples x; to get the assigned class
%            use y_class=sign(y).
%
%  INPUT:     x=[x1,x2,...,xN] inputs points to classify; xk is nx1 real
%             vector;
%   SVC PARAMETERS:
%              ap,an: alphas of positive and negative support vectors
%              svp/svn: positive/negative support vectors (0<a<c)
%              mvp/mvn: positive/negative margin vectors (a=c)
%              C is the sofmargin parameter; 
%              FOR HARD-MARGIN CLASSIFICATION set C=Inf and mvp=[]; mvn=[];
%              w0 is the bias parameter
%             'kernel' is the type of kernel used;
%   Options for Kernel function:
%              1) kernel='linear' --> K(u,v)=u*v' 
%              2) kernel='poly'   --> K(u,v)=(u*v'+1)^p1 
%              3) kernel='rbf'    --> K(u,v)=exp(-(u-v)*(u-v)'/(2*p1^2)) 
%              4) kernel='sigmoid'--> K(u,v)=tanh(p1*u*v'/length(u)+p2) 
%              kpar defines corresponding kernel parameters, e.g kpar=[p1, p2]; 
%
%  OUTPUT:     y=[y1,y2,...,yN] assigned classes yk= 1 or -1; 
%
%              ----------

%   written by A. Sideris, 10/19/04, updated: 2/18/2016 for MAE 277

if isinf(C)
    B=0;
else
    B=C;
end

y=kprod(x,svp,kernel,kpar)*ap-kprod(x,svn,kernel,kpar)*an...
    +sum(kprod(x,mvp,kernel,kpar),2)*B-sum(kprod(x,mvn,kernel,kpar),2)*B + w0;




function k=kprod(u,v,ker,kpar)

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

return
