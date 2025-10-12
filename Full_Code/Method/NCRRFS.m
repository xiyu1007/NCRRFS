classdef NCRRFS
    properties
        % 基本参数
        name = 'NCRRFS';
        max_iter = 100;
        tol = 1e-3;
        toleration = 1e-9;
        iter_tol = 1e-3;

        verbose = 0;
        keyboard = 0;
        Loss = [];
        Err = [];
        Err2 = [];

        start_time=1;
        runtime = 0;
        
        % 超参数
        alpha;   % 标签一致性系数
        beta;    % W稀疏性系数
        lambda;   % 视图一致性系数
        eta =0 ;
        gamma1;
        gamma2 = 0;
        phi;
        
        % 数据维度
        n;   % 样本数
        M;     % 视图数
        d;        % 各视图特征维度[]
        c;  
        
        % 优化变量
        W;           % 各视图投影矩阵 cell
        S;           % 各视图隐表示 cell
        Z;           % 共识表示
        E;           % 权重
        L;
        K;
        k = 10;
        % Lagrange multipliers
        U1 = 0;
        U2;
        mu = 1;
        mu_max = 1e6;
        delta = 1.1;
        epsilon = 1e-5;

    end
    
    methods
        function obj = NCRRFS()
        end

        % Set parameters
        function obj = setParams(obj, params)
            obj.alpha = params(1);
            obj.beta = params(2);
            obj.lambda = params(3);
            obj.eta = params(4);
            obj.gamma1 = params(5);
            obj.gamma2 = params(6);
        end

        % Initialize
        function obj = init(obj,n, c, M, d)
            obj.M = M;
            obj.n = n;
            obj.d = d;
            obj.c = c; 
           
            obj.E = cell(M, 1);
            obj.W = cell(M, 1);
            obj.U2 = cell(M, 1);
            for m = 1:M
                obj.W{m} = zeros(d(m), c);
                obj.E{m} = zeros(c, n);
                obj.U2{m} = zeros(c, n);
            end
            obj.S = zeros(n,n);
            obj.L = cell(M, 1);
            obj.Z = zeros(n, n);
            obj.U1 = zeros(n, n);

            obj.start_time = tic;
        end
   
        function [f,df] = SCAD(~,x,gamma)
            a = 3.7;

            df = (a*gamma .* sign(x) - x) / (a-1);
            id = abs(x) > a*gamma;
            df(id,1) = 0;
            id = abs(x) <= gamma;
            df(id,1) = gamma .* sign(x(id,1));
            
            f = (2*a*gamma .* abs(x) - x.^2 - gamma^2) / (2*(a-1));
            id = abs(x) > a*gamma;
            f(id,1) = gamma^2*(a+1) / 2;
            id = abs(x) <= gamma;
            f(id,1) = gamma .*abs(x(id,:));
        end

        function Z = updateZ(obj)
            U1 = obj.U1;
            S = obj.S;
            mu = obj.mu;
            ref = obj.lambda / mu;
            
            Phi = S - (U1/mu);

            [U, sigmoid, V] = svd(Phi,"econ");  %只包含与非零奇异值对应的列/行。
            singular_values = diag(sigmoid);
            idx = singular_values > ref;  % 找出大于 1/μ 的索引
            Sigma0 = diag(singular_values(idx) - ref);  % 减去阈值
            U0 = U(:, idx);
            V0 = V(:, idx);
            Z = U0 * Sigma0 * V0';
        end

        function S = updateS(obj,X)
            W = obj.W;
            E = obj.E;
            U2 = obj.U2;
            U1 = obj.U1;
            mu = obj.mu;
            Z = obj.Z;

            term1 = eye(obj.n);
            term2 = Z + U1/mu;
            for m=1:obj.M
                WTX = W{m}' * X{m};
                Cv = W{m}' * X{m} - E{m} + (U2{m} / mu);
                term1 = term1 + WTX' * WTX;
                term2 = term2 + WTX' * Cv;
            end
            S = term1 \ term2;
        end

        function W = updateW(obj,X, Y)
            g1 = obj.gamma1;
            alpha = obj.alpha;
            eta = obj.eta;
            U2 = obj.U2;
            S = obj.S;
            E = obj.E;
            W = obj.W;
            L = obj.L;

            I = eye(obj.n);
            IS = (I - S);
            mu = obj.mu;
            mu2 = mu/2;
            IIS = (I + mu2*IS*(IS'));
            for m=1:obj.M
                Xv = X{m};
                Wv = W{m};
                Ev = E{m};

                XIS = Xv*IS;
                ep = obj.tol *eye(obj.d(m));
                term1 = Xv*IIS*Xv' + eta*L{m} + ep;
                temp = Ev - (U2{m}/mu);
                EY = Ev + Y;
                term2 = mu2*XIS*temp' + Xv*EY';

                norm2row = vecnorm(Wv,2,2);
                [~,df] = obj.SCAD(norm2row,g1);
                Dv = diag(df ./ (2*norm2row +obj.iter_tol) );
                term1 = term1 + alpha*Dv;

                W{m} = term1 \ term2;
            end
        end
        
        function E = updateE(obj,X,Y)
            beta = obj.beta;
            g2 = obj.gamma2;
            U2 = obj.U2;
            S = obj.S;
            E = obj.E;
            W = obj.W;

            mu = obj.mu;
            mu2 = mu/2;
            I = eye(obj.n);
            for m=1:obj.M
                Xv = X{m};
                Wv = W{m};
                Ev = E{m};
                WTX = Wv'*Xv;

                temp = WTX - WTX*S + (U2{m}/obj.mu);
                term1 = mu2*temp + (WTX - Y);
                term2 = (mu2 + 1 + beta)*I;

                norm2row = vecnorm(Ev',2,2);
                [~,df] = obj.SCAD(norm2row,g2);
                Dv = diag(df ./ (2*norm2row +obj.iter_tol) );
                term2 = term2 + beta*Dv;
                E{m} = term1 / term2;
            end
        end

        function L = updateL(obj,X)
            L = obj.L;
            X = TransposeXY(X);
            S = getCovSimilarity(X, [], 1, obj.k);

            for m=1:obj.M
                G = S{m};
                G = (G + G')/2;
                D = sum(G,2);
                L{m} = diag(D) - (G);
            end
        end

        function [f,df] = Fun(obj, X, Y)
            eta = obj.eta;
            E = obj.E;
            W = obj.W;
            g1 = obj.gamma1;
            g2 = obj.gamma2;
            alpha = obj.alpha;
            beta = obj.beta;
            lambda = obj.lambda;
            L = obj.L;
            
            f1 = 0;
            f2 = 0;
            f3 = 0;
            f4 = sum(svd(obj.S,'econ'));
            f5 = 0; 
            fw = 0;
            for m=1:obj.M
                WTXE = W{m}'*X{m} - E{m};
                f1 = f1 + norm(WTXE - Y,'fro')^2;

                norm2W = vecnorm(W{m},2,2);
                f2 = f2 + sum(obj.SCAD(norm2W,g1));

                norm2E = vecnorm(E{m}',2,2);
                f3 = f3 + sum(obj.SCAD(norm2E,g2));
                f3 = f3 + norm(E{m},'fro')^2;

                f5 = f5 + trace(W{m}'*L{m}*W{m});
                fw = fw + obj.tol*norm(W{m},'fro')^2;
            end
            f = f1 + alpha*f2 + beta*f3 + lambda*f4 + eta*f5 + fw;
            df = 0;
        end

        function [f,df] = LFun(obj, X, Y)
            eta = obj.eta;
            E = obj.E;
            W = obj.W;
            g1 = obj.gamma1;
            g2 = obj.gamma2;
            alpha = obj.alpha;
            beta = obj.beta;
            lambda = obj.lambda;
            L = obj.L;
            
            f1 = 0;
            f2 = 0;
            f3 = 0;
            f4 = sum(svd(obj.Z,'econ'));
            f5 = 0; 
            fw = 0;
            fzs = (obj.mu/2)*norm(obj.Z - obj.S + (obj.U1/obj.mu),'fro')^2;
            fwe = 0;
            for m=1:obj.M
                WTX = W{m}'*X{m};
                WTXE = WTX - E{m};
                f1 = f1 + norm(WTXE - Y,'fro')^2;

                norm2W = vecnorm(W{m},2,2);
                f2 = f2 + sum(obj.SCAD(norm2W,g1));

                norm2E = vecnorm(E{m}',2,2);
                f3 = f3 + sum(obj.SCAD(norm2E,g2));
                f3 = f3 + norm(E{m},'fro')^2;

                f5 = f5 + trace(W{m}'*L{m}*W{m});
                fw = fw + obj.tol*norm(W{m},'fro')^2;
                fwe = fwe + (obj.mu/2)*norm(WTX - WTX*obj.S - E{m} + (obj.U2{m}/obj.mu),'fro')^2;
            end
            f = f1 + alpha*f2 + beta*f3 + lambda*f4 + eta*f5 + fw + fzs + fwe;
            df = 0;
        end
    
        function [U1,U2,mu] = updateLagrange(obj,X)
            W = obj.W;
            E = obj.E;
            U1 = obj.U1 + obj.mu*(obj.Z - obj.S);
            U2 = obj.U2;
            for m=1:obj.M
                U2{m} = U2{m} + obj.mu* (W{m}'*X{m} - W{m}'*X{m}*obj.S - E{m});
            end
            mu = min(obj.delta*obj.mu,obj.mu_max);
        end


        function obj = run(obj,X,Y,param,seed)
            warning('off');
            % obj.verbose = 1;

            % rng(seed);
            [n, c, M, d] = getDataInfo(X,Y);
            [X, Y] = TransposeXY(X, Y);
            max_iter = obj.max_iter;

            obj = setParams(obj, param);
            obj = obj.init(n, c, M, d);

            obj.L = obj.updateL(X);
            % [fo,~] = obj.Fun(X, Y);

            for iter = 1:max_iter
                obj.W = obj.updateW(X,Y);
                obj.Z = obj.updateZ();
                obj.S = obj.updateS(X);
                obj.E = obj.updateE(X,Y);
                [obj.U1,obj.U2,obj.mu] = obj.updateLagrange(X);

                [f,~] = obj.Fun(X, Y);
                obj.Loss(iter) = f;

                C1 = norm(obj.Z - obj.S,Inf);
                C2 = 0;
                for m=1:M
                    term = obj.W{m}'*X{m} - obj.W{m}'*X{m}*obj.S - obj.E{m};
                    C2 = max(C2,norm(term,Inf));
                end
                obj.Err(iter) = C2;
                obj.Err2(iter) = C1;
                % rate = abs(fo -f) / abs(fo);
                if C2 < obj.iter_tol && C1 < obj.iter_tol  && iter > 8 %  && rate < obj.iter_tol
                    break;
                end
                % fo = f;
            end

            if obj.verbose
                figure
                % 绘制第一条曲线
                linew = 1.2;
                plot(obj.Loss ./ max(obj.Loss),'LineWidth',linew)
                hold on 
                plot(obj.Err ./ max(obj.Err),'LineWidth',linew)
                plot(obj.Err2 ./ max(obj.Err2),'LineWidth',linew)
                legend show;
                legend({'$Loss$','$||W^{v^T} X^v - W^{v^T} X^v S - E^v||_\infty$', '$||Z - S||_\infty$'}, ...
                    'Interpreter', 'latex', 'FontSize', 13,'Location', 'best');
                yticks(0:0.2:1);
                % 获取当前坐标轴范围
                x_limits = xlim;
                y_limits = ylim;
                % 设置字体
                set(gca, 'FontSize', 14, 'FontName', 'Times New Roman');
                % 恢复坐标轴范围
                xlim(x_limits);
                ylim(y_limits);
                % fig = gcf;
                % exportgraphics(fig, ['Analyze\Fig\','Loss_A1.pdf'], 'ContentType', 'vector');
                % close;
            end
        end

        function parameter = init_param(~,fix)
            if nargin < 2
                fix = [Inf, Inf, Inf, Inf, Inf Inf];
            end
            alphaSpace = logspace(-3,3,7);
            betaSpace = logspace(-3,3,7);
            lambdaSpace = logspace(-3,3,7);
            eatSpace = logspace(-3,3,7);

            g1Space = [0.01, 0.1, 0.5, 0.7, 1, 2, 3  5, 10, 30];
            g2Space = [0.01, 0.1, 0.5, 0.7, 1, 2, 3  5, 10, 30];
            if fix(1) ~= Inf
                alphaSpace = fix(1);
            end
            if fix(2) ~= Inf
                betaSpace = fix(2);
            end
            if fix(3) ~= Inf
                lambdaSpace = fix(3);
            end
            if fix(4) ~= Inf
                eatSpace = fix(4);
            end
            if fix(5) ~= Inf
                g1Space = fix(5);
            end
            if fix(6) ~= Inf
                g2Space = fix(6);
            end

            paramSpace = {alphaSpace, betaSpace, lambdaSpace,eatSpace, g1Space,g2Space};
            parameter = combvec(paramSpace{:})';
            parameter = sortrows(parameter, 'ascend');
        end

    end
end