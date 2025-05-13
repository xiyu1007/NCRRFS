function K = mk_kernel(X1,varargin)
    % X1: 测试集样本 (N_test x D)
    % X2: 训练集样本 (N_train x D)


    modalities = numel(X1);

    p = inputParser;
    addRequired(p, 'X1');
    addOptional(p, 'X2', X1, @(x) iscell(x) && numel(x) == numel(X1));
    addOptional(p, 'weights', ones(1, numel(X1)) / numel(X1));
    addOptional(p, 'kernels', repmat({'rbf'}, 1, numel(X1)), @(x) iscell(x));
    addOptional(p, 'kernel_params', repmat({{}}, 1, numel(X1)), @(x) iscell(x));
    addOptional(p, 'z_score', 1, @(x) isnumeric(x));

    % 解析输入参数
    parse(p, X1,varargin{:});
    X2 = p.Results.X2;
    z_score = p.Results.z_score;
    weights = p.Results.weights;
    if isempty(weights)
        weights = ones(1, numel(X1)) ./ numel(X1);
    end
    kernels = p.Results.kernels;
    if isscalar(kernels)
        kernels = repmat(kernels, 1, modalities);
    end
    kernel_params = p.Results.kernel_params;
    if isscalar(kernel_params)
        kernel_params = repmat(kernel_params, 1, modalities);
    end

        % 样本数量
    N1 = size(X1{1}, 1);
    N2 = size(X2{1}, 1);
    K = zeros(N1, N2); % 初始化核矩阵

    for i = 1:modalities
        if z_score
            scaler = Scaler();
            [x2,scaler] = scaler.fit_transform(X2{i});
            x1 = scaler.transform(X1{i});
            K_modality = calculate_kernel(x1, x2, kernels{i}, kernel_params{i});
        else
            K_modality = calculate_kernel(X1{i}, X2{i}, kernels{i}, kernel_params{i});
        end

        K = K + weights(i) * K_modality;
    end
end
