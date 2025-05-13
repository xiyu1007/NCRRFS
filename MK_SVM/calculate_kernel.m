function K = calculate_kernel(X, Y, kernel_type, params)
    % 通用核矩阵计算函数
    % 
    % 输入：
    %   X          - 数据矩阵 (N x D) 
    %   Y          - 数据矩阵 (M x D) 
    %   kernel_type- 核函数类型，例如 'rbf', 'linear', 'polynomial'
    %   params     - 核函数的参数 cell 数组
    %
    % 输出：
    %   K          - 核矩阵 (N x M)

    % 核函数计算
    switch kernel_type
        case 'linear'
            K = X * Y';
            
        case 'rbf'
            % 默认gamma
            if isempty(params)
                % gamma = 1 / mean(pdist(X).^2);  
                gamma = 1 / (size(X, 2) * var(X(:)));
            else
                gamma = params{1};
            end
            sq_norm_X = sum(X.^2, 2);
            sq_norm_Y = sum(Y.^2, 2);
            dist_sq = bsxfun(@plus, sq_norm_X, sq_norm_Y') - 2 * (X * Y');
            K = exp(-gamma * dist_sq);
            
        case 'polynomial'
            % 默认参数: gamma=1, coef0=1, degree=3
            if isempty(params)
                gamma = 1; coef0 = 1; degree = 3;
            else
                gamma = params{1}; 
                coef0 = numel(params) > 1 && params{2} || 1;
                degree = numel(params) > 2 && params{3} || 3;
            end
            K = (gamma * (X * Y') + coef0).^degree;
            
        otherwise
            error('不支持的核函数类型：%s', kernel_type);
    end
end
