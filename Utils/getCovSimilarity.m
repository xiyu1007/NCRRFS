function S = getCovSimilarity(X,Y, dim, k, method)
    % getCovSimilarity - 计算样本或特征之间的相似度矩阵
    % 
    % 输入:
    %   X      : n × d 矩阵
    %   Y      : n × c 矩阵
    %   dim    : 1 -> 特征间相似度 (列)，2 -> 样本间相似度 (行)
    %   method : 'pearson' 或 'cosine'
    %
    % 输出:
    %   S : 相似度矩阵（对称，归一化到 [0,1]）
    if nargin < 2
        Y = [];  % 默认 Y 为空
    end
    if size(Y,1) < size(Y,2) 
        [X, Y] = TransposeXY(X, Y);
    end
    if nargin < 3
        dim = 1;
    end
    if nargin < 4
        k = [];
    end
    if nargin < 5
        method = 'pearson';
    end

    % arguments
    %     X double
    %     dim (1,1) {mustBeMember(dim,[1,2])}
    %     method (1,:) char {mustBeMember(method, {'pearson', 'cosine'})}
    % end
    M = numel(X);
    S = cell(1,M);

    for m=1:M
        Xm = X{m};
        % 如果要计算特征间相似度，需要转置数据（变为列向量）

        % % 初始化标准化器
        % scaler = Scaler(1);
        [Xm, ~] = Scaler().fit_transform(Xm);
        if dim == 1
            Xm = Xm';  % 变成 d × n
        end
    
        % 样本或特征个数
        n = size(Xm, 1);
        Sm = zeros(n, n);

        zero_columns = all(Xm == 0, 2);
        switch method
            case 'cosine'
                % 每行归一化成单位向量
                X_norm = Xm ./ vecnorm(Xm, 2, 2);
                Sm = X_norm * X_norm';  % 余弦相似度
                % Sm = abs(Sm);              % 映射到 [0, 1]
                Sm = max(min(Sm, 1), 0); % 限制在 [0,1]
    
            case 'pearson'
                R = corr(Xm');          % corr计算列之间相关性 需要变量为列 → 所以转置
                % Sm = (R + 1) / 2;              % 映射到 [0, 1]
                % Sm = abs(R);              % 映射到 [0, 1]
                Sm = max(min(R, 1), 0); % 限制在 [0,1]
        end
        if ~isempty(Y) && size(Y,1) == n
            mask = Y(:, 1) == Y(:, 1)';  % 类别相等的掩码
            Sm = Sm .* mask;
        end
        Sm(zero_columns,:) = 0;
        Sm(:,zero_columns) = 0;
        Sm(logical(eye(n))) = 1;
        if ~isempty(k)
            % Sm(logical(eye(n))) = 0;
            for j = 1:n  % m 是矩阵 sim 的列数
                % 获取当前列的相似度值
                col = Sm(:, j);
                % 对当前列的相似度值排序，得到排序后的索引（降序排序）
                [~, ids] = sort(col, 'descend');
                % 获取前k个最相似值的索引
                top_k_indices = ids(1:k);
                % 创建一个与原列相同大小的全零列
                new_col = zeros(length(col), 1);
                % 将前k个最相似值置为原值
                new_col(top_k_indices) = col(top_k_indices);
                % 更新原始矩阵
                Sm(:, j) = new_col;
            end
            Sm = (Sm + Sm') / 2;
        end
        S{m} = Sm;
        % drawSim(Sm);
    end
end
