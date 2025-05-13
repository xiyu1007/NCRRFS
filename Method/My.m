classdef My
    % My class is designed to perform multi-view learning with specific optimization methods.
    % The class optimizes multi-view data by minimizing a loss function that considers 
    % view-specific projections, similarity matrices, and consensus structures.
    %
    % Properties:
    %   name        : Name of the model.
    %   max_iter    : Maximum number of iterations.
    %   tol         : Tolerance for other parameters.
    %   iter_tol    : Iteration tolerance for convergence criteria.
    %   Loss        : Array to store the loss value at each iteration.
    %   Err         : Array to store error values at each iteration.
    %   Err2        : Array to store second error values at each iteration.
    %   start_time  : Start time for tracking the runtime.
    %   runtime     : Total runtime of the optimization process.
    %   alpha       : Regularization parameter for the first term.
    %   beta        : Regularization parameter for the second term.
    %   lambda      : Regularization parameter for the third term.
    %   eta         : Regularization parameter for the fourth term.
    %   gamma1      : Regularization parameter for sparsity in first term.
    %   gamma2      : Regularization parameter for sparsity in second term.
    %   n           : Number of samples in the dataset.
    %   M           : Number of views.
    %   d           : Dimensions of each view’s feature space.
    %   c           : Number of clusters or categories.
    %   W           : Cell array of view-specific projection matrices.
    %   S           : Matrix of shared low-dimensional representations.
    %   Z           : Matrix for consensus representation across all views.
    %   E           : Cell array of error matrices.
    %   L           : Cell array of view-specific Laplacian matrices.
    %   k           : Number of nearest neighbors for similarity computation.
    %   U1          : Lagrange multiplier for consensus.
    %   U2          : Cell array of Lagrange multipliers for error terms.
    %   mu          : Regularization multiplier for constraints.
    %   mu_max      : Maximum allowable value for the Lagrange multiplier.
    %   delta       : Multiplier factor for updating the Lagrange multiplier.
    
    properties
        name = 'My';
        max_iter = 100;
        tol = 1e-6;
        iter_tol = 1e-3;

        Loss = [];
        Err = [];
        Err2 = [];

        start_time = 1;
        runtime = 0;
        
        alpha;  
        beta;   
        lambda; 
        eta = 0 ;
        gamma1 = 0;
        gamma2 = 0;
        
        % 数据维度 (Data Dimensions)
        n;   % 样本数 (Number of samples)
        M;     % 视图数 (Number of views)
        d;        % 各视图特征维度 (Feature dimensions of each view)
        c;  
        
        % 优化变量 (Optimization Variables)
        W;           % 各视图投影矩阵 cell (Cell array of view-specific projection matrices)
        S;           % 各视图隐表示 cell (Cell array of hidden representations for each view)
        Z;           % 共识表示 (Consensus representation matrix)
        E;           % 权重 (Error matrices for each view)
        L;           % Laplacian matrices for each view
        k = 10;      % k-nearest neighbors for similarity computation
        
        % Lagrange multipliers
        U1 = 0;
        U2;
        mu = 1;
        mu_max = 1e6;
        delta = 1.1;
    end
    
    methods
        function obj = My()
            % Constructor for the My class.
            % Initializes the object with default values for properties.
        end

        % Set parameters for optimization
        function obj = setParams(obj, params)
            % Sets the values for alpha, beta, lambda, eta, gamma1, and gamma2 from a parameter list.
            % Inputs:
            %   params - A vector of parameter values.
            % Outputs:
            %   obj - Updated My object with new parameters.
            
            obj.alpha = params(1);
            obj.beta = params(2);
            obj.lambda = params(3);
            obj.eta = params(4);
            obj.gamma1 = params(5);
            obj.gamma2 = params(6);
        end

        % Initialize optimization variables
        function obj = init(obj, n, c, M, d)
            % Initializes variables for the optimization process.
            % Inputs:
            %   n - Number of samples
            %   c - Number of clusters
            %   M - Number of views
            %   d - Dimensions of each view's feature space
            % Outputs:
            %   obj - Initialized My object.

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
            obj.S = zeros(n, n);
            obj.L = cell(M, 1);
            obj.Z = zeros(n, n);
            obj.U1 = zeros(n, n);

            obj.start_time = tic;
        end

        % SCAD penalty function
        function [f, df] = SCAD(~, x, gamma)
            % SCAD function computes the smooth penalty function and its gradient.
            % Inputs:
            %   x     - The input vector for which to compute the penalty.
            %   gamma - A regularization parameter.
            % Outputs:
            %   f     - The penalty function value.
            %   df    - The gradient of the penalty function.
            
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
            f(id,1) = gamma .* abs(x(id,:));
        end

        % Update the consensus representation (Z)
        function Z = updateZ(obj)
            % Updates the consensus representation matrix Z using the current value of U1 and S.
            % Outputs:
            %   Z - Updated consensus representation matrix.
            
            U1 = obj.U1;
            S = obj.S;
            mu = obj.mu;
            ref = obj.lambda / mu;
            
            Phi = S - (U1/mu);
            [U, sigmoid, V] = svd(Phi,"econ");  
            singular_values = diag(sigmoid);
            idx = singular_values > ref;  
            Sigma0 = diag(singular_values(idx) - ref);
            U0 = U(:, idx);
            V0 = V(:, idx);
            Z = U0 * Sigma0 * V0';
        end

        % Update the shared low-dimensional representations (S)
        function S = updateS(obj, X)
            % Updates the shared low-dimensional representations matrix S.
            % Inputs:
            %   X - Cell array of input data for each view.
            % Outputs:
            %   S - Updated shared representations matrix.
            
            W = obj.W;
            E = obj.E;
            U2 = obj.U2;
            U1 = obj.U1;
            mu = obj.mu;
            Z = obj.Z;

            term1 = eye(obj.n);
            term2 = Z + U1/mu;
            for m = 1:obj.M
                WTX = W{m}' * X{m};
                Cv = W{m}' * X{m} - E{m} + (U2{m} / mu);
                term1 = term1 + WTX' * WTX;
                term2 = term2 + WTX' * Cv;
            end
            S = term1 \ term2;
        end
        
        % Update the projection matrices (W)
        function W = updateW(obj, X, Y)
            % Updates the projection matrices for each view.
            % Inputs:
            %   X - Cell array of input data for each view.
            %   Y - Target output data.
            % Outputs:
            %   W - Updated projection matrices for each view.
            
            g1 = obj.gamma1;
            alpha = obj.alpha;
            eta = obj.eta;
            U2 = obj.U2;
            S = obj.S;
            E = obj.E;
            W = obj.W;
            L = obj.L;

            I = eye(obj.n);
            mu = obj.mu;
            mu2 = mu/2;
            for m = 1:obj.M
                Xv = X{m};
                Wv = W{m};
                Ev = E{m};

                XIS = Xv*(I - S);
                ep = obj.tol *eye(obj.d(m));
                term1 = Xv*Xv' + mu2*(XIS*XIS') + eta*L{m}+ ep;
                temp = Ev - (U2{m}/mu);
                EY = Ev + Y;
                term2 = mu2*XIS*temp' + Xv*EY';

                norm2row = vecnorm(Wv,2,2);
                [~,df] = obj.SCAD(norm2row,g1);
                Dv = diag(df ./ (2*norm2row + obj.iter_tol));
                term1 = term1 + alpha*Dv;

                W{m} = term1 \ term2;
            end
        end
        
        % Update the error matrices (E)
        function E = updateE(obj, X, Y)
            % Updates the error matrices for each view.
            % Inputs:
            %   X - Cell array of input data for each view.
            %   Y - Target output data.
            % Outputs:
            %   E - Updated error matrices for each view.
            
            beta = obj.beta;
            g2 = obj.gamma2;
            U2 = obj.U2;
            S = obj.S;
            E = obj.E;
            W = obj.W;
            mu = obj.mu;
            mu2 = mu/2;
            I = eye(obj.n);
            for m = 1:obj.M
                Xv = X{m};
                Wv = W{m};
                Ev = E{m};
                WTX = Wv'*Xv;

                temp = WTX - WTX*S + (U2{m}/obj.mu);
                term1 = mu2*temp + (WTX - Y);
                term2 = (mu2 + 1 + beta)*I;

                norm2row = vecnorm(Ev',2,2);
                [~,df] = obj.SCAD(norm2row,g2);
                Dv = diag(df ./ (2*norm2row + obj.iter_tol));
                term2 = term2 + beta*Dv;
                E{m} = term1 / term2;
            end
        end

        % Update the Laplacian matrices (L)
        function L = updateL(obj, X)
            % Updates the Laplacian matrices for each view.
            % Inputs:
            %   X - Cell array of input data for each view.
            % Outputs:
            %   L - Updated Laplacian matrices for each view.
            
            L = obj.L;
            X = TransposeXY(X);
            S = getCovSimilarity(X, [], 1, obj.k);
            for m = 1:obj.M
                G = S{m};
                G = (G + G')/2;
                D = sum(G,2);
                L{m} = diag(D) - (G);
            end
        end

        % Compute the main loss function and its gradient
        function [f, df] = Fun(obj, X, Y)
            % Computes the main loss function with respect to X and Y.
            % Inputs:
            %   X - Cell array of input data for each view.
            %   Y - Target output data.
            % Outputs:
            %   f  - The value of the loss function.
            %   df - The gradient of the loss function (not used in the current implementation).
            
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
            for m = 1:obj.M
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

        % Compute the auxiliary loss function for the Lagrange multipliers
        function [f, df] = LFun(obj, X, Y)
            % Computes the auxiliary loss function used for Lagrange multiplier updates.
            % Inputs:
            %   X - Cell array of input data for each view.
            %   Y - Target output data.
            % Outputs:
            %   f  - The value of the auxiliary loss function.
            %   df - The gradient of the auxiliary loss function (not used in the current implementation).
            
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
            for m = 1:obj.M
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

        % Update the Lagrange multipliers (U1, U2, mu)
        function [U1, U2, mu] = updateLagrange(obj, X)
            % Updates the Lagrange multipliers U1, U2, and the regularization parameter mu.
            % Inputs:
            %   X - Cell array of input data for each view.
            % Outputs:
            %   U1 - Updated Lagrange multiplier for consensus.
            %   U2 - Updated Lagrange multipliers for error terms.
            %   mu - Updated value for the regularization parameter.
            
            W = obj.W;
            E = obj.E;
            U1 = obj.U1 + obj.mu*(obj.Z - obj.S);
            U2 = obj.U2;
            for m = 1:obj.M
                U2{m} = U2{m} + obj.mu* (W{m}'*X{m} - W{m}'*X{m}*obj.S - E{m});
            end
            mu = min(obj.delta*obj.mu, obj.mu_max);
        end

        function obj = run(obj, X, Y, param, seed)
            % Description:
            %   The main optimization loop that trains the model. Initializes all
            %   required variables, and iteratively updates W, Z, S, E, and the 
            %   Lagrange multipliers until convergence criteria are met or the
            %   maximum number of iterations is reached.
            %
            % Inputs:
            %   - X     : input data (cell array for each view)
            %   - Y     : label matrix or target projection
            %   - param : parameter vector [alpha, beta, lambda, eta, gamma1, gamma2]
            %   - seed  : random seed for reproducibility
            %
            % Outputs:
            %   - obj   : trained model object containing optimized variables and loss history
            warning('off');
            rng(seed);
            [n, c, M, d] = getDataInfo(X, Y);
            [X, Y] = TransposeXY(X, Y);
            max_iter = obj.max_iter;

            obj = setParams(obj, param);
            obj = obj.init(n, c, M, d);

            obj.L = obj.updateL(X);
            
            for iter = 1:max_iter
                obj.W = obj.updateW(X, Y);
                obj.Z = obj.updateZ();
                obj.S = obj.updateS(X);
                obj.E = obj.updateE(X, Y);
                [obj.U1, obj.U2, obj.mu] = obj.updateLagrange(X);

                [f, ~] = obj.Fun(X, Y);
                obj.Loss(iter) = f;

                % Compute stopping conditions
                C1 = norm(obj.Z - obj.S, Inf);
                C2 = 0;
                for m = 1:M
                    term = obj.W{m}' * X{m} - obj.W{m}' * X{m} * obj.S - obj.E{m};
                    C2 = max(C2, norm(term, Inf));
                end
                obj.Err(iter) = C2;
                obj.Err2(iter) = C1;

                if C2 < obj.iter_tol && C1 < obj.iter_tol && iter > 8
                    break;
                end
            end

            % figure
            % plot(obj.Loss ./ max(obj.Loss))
            % hold on 
            % plot(obj.Err ./ max(obj.Err))
            % plot(obj.Err2 ./ max(obj.Err2))
            % legend show;
            % legend({'$Loss$','$|W^{v^T} X^v - W^{v^T} X^v S^v - E^v||_\infty$', '$|Z^v - S^v||_\infty$'}, ...
            %     'Interpreter', 'latex');
            % drawSim(obj.S,Y);
            % drawSim(obj.Z,Y);

        end

        function parameter = init_param(~, fix)
            % Description:
            %   Generates a grid of candidate hyperparameter combinations for
            %   model selection or cross-validation. If fixed values are provided,
            %   those are used instead of grid search values.
            %
            % Inputs:
            %   - fix : vector of fixed parameters [alpha, beta, lambda, eta, gamma1, gamma2]
            %          Use Inf to indicate a parameter should be searched over a range
            %
            % Outputs:
            %   - parameter : matrix of parameter combinations (rows correspond to different sets)
            if nargin < 2
                fix = [Inf, Inf, Inf, Inf, Inf, Inf];
            end

            alphaSpace  = logspace(-3, 3, 7);
            betaSpace   = logspace(-3, 3, 7);
            lambdaSpace = logspace(-3, 3, 7);
            eatSpace    = logspace(-3, 3, 7);
            g1Space     = [0.01, 0.1, 0.5, 0.7, 1, 2, 3, 5, 10, 30];
            g2Space     = [0.01, 0.1, 0.5, 0.7, 1, 2, 3, 5, 10, 30];

            if fix(1) ~= Inf, alphaSpace  = fix(1); end
            if fix(2) ~= Inf, betaSpace   = fix(2); end
            if fix(3) ~= Inf, lambdaSpace = fix(3); end
            if fix(4) ~= Inf, eatSpace    = fix(4); end
            if fix(5) ~= Inf, g1Space     = fix(5); end
            if fix(6) ~= Inf, g2Space     = fix(6); end

            paramSpace = {alphaSpace, betaSpace, lambdaSpace, eatSpace, g1Space, g2Space};
            parameter = combvec(paramSpace{:})';
            parameter = sortrows(parameter, 'ascend');
        end
    end
end