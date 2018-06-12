function [Z,S,obj] = AWLRR3(X,Z_ini,lambda1,lambda2,lambda3,max_iter,Ctg,miu,rho)
% The code is written by Jie Wen, 
% if you have any problems, please don't hesitate to contact me: wenjie@hrbeu.edu.cn 
% If you find the code is useful, please cite the following reference:
% J. Wen, B. Zhang, Y. Xu, J. Yang, and N. Han, 
% Adaptive Weighted Nonnegative Low-Rank Representation, 
% Pattern Recognition, 2018.

max_miu = 1e8;
tol  = 1e-6;
tol2 = 1e-2;
C1 = zeros(size(X));
C2 = zeros(size(Z_ini));
C3 = zeros(size(Z_ini));
S = ones(size(X));
distX = L2_distance_1(X,X);
D = lambda3*distX;
for iter = 1:max_iter
    if iter == 1
        Z = Z_ini;
        U = Z_ini;
        E = X-X*Z;
    end
    Z_old = Z;
    U_old = U; 
    E_old = E;
    S_old = S;

    % ------------ S ------------- %
    S_linshi = -(E.^2)/lambda1;
    S = zeros(size(S_linshi));
    for ii = 1:size(E,2)
        S(:,ii) = EProjSimplex(S_linshi(:,ii));
    end
    % --------- E -------- %
    G = X-X*Z+C1/miu;
    E = (miu*G)./(miu+2*S);
    % -------- Z ------------ %
    M1 = X-E+C1/miu;
    M2 = U-C2/miu;
    Z = Ctg*(X'*M1+M2-D/miu);
    Z = Z - diag(diag(Z));
    for ii = 1:size(Z,2)
        idx = 1:size(Z,2);
        idx(ii) = [];
        Z(ii,idx) = EProjSimplex_new(Z(ii,idx));
    end
    % ------------ U ------------ %
    tempU = Z+C2/miu;
    [AU,SU,VU] = svd(tempU,'econ');
    AU(isnan(AU)) = 0;
    VU(isnan(VU)) = 0;
    SU(isnan(SU)) = 0;
    SU = diag(SU);    
    SVP = length(find(SU>lambda2/miu));
    if SVP >= 1
        SU = SU(1:SVP)-lambda2/miu;
    else
        SVP = 1;
        SU = 0;
    end
    U = AU(:,1:SVP)*diag(SU)*VU(:,1:SVP)';
 
    % ------ C1 C2 miu ---------- %
    L1 = X-X*Z-E;
    L2 = Z-U; 
    C1 = C1+miu*L1;
    C2 = C2+miu*L2;
    
    LL1 = norm(Z-Z_old,'fro');
    LL2 = norm(U-U_old,'fro');
    LL3 = norm(E-E_old,'fro');
    LL4 = norm(S-S_old,'fro');
    SLSL = max(max(LL1,LL2),LL3)/norm(X,'fro');
    if miu*SLSL < tol2
        miu = min(rho*miu,max_miu); 
    end
    stopC = (norm(L1,'fro')+norm(L2,'fro'))/norm(X,'fro');
    if stopC < tol
        iter
        break;
    end
    obj(iter) = stopC;   
end
end




    