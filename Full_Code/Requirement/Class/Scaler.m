classdef Scaler
    % X ：n*d
    properties
        % 存储训练数据的均值和标准差
        meanX
        stdX
        epsilon = 1e-8  % 防止标准差为零时除以零
        dim = 1; % 默认特征标准化 n*d
    end
    
    methods
        function obj = Scaler(dim)
            if nargin > 0
                obj.dim = dim;
            end
        end
        
        % fit方法：计算并存储均值和标准差
        function obj = fit(obj, X)
            % 计算数据的均值和标准差
            obj.meanX = mean(X, obj.dim); % 1*d
            obj.stdX = std(X, 0, obj.dim);
            % zero_col = (stdX == 0);
            % stdX(zero_col) = obj.epsilon;
            % obj.stdX = stdX;
        end
        
        % transform方法：使用存储的均值和标准差来标准化数据
        function X_scaled = transform(obj, X)
            % 使用记录的均值和标准差对数据进行标准化
            X_scaled = (X - obj.meanX) ./ (obj.stdX + obj.epsilon);
        end
        
        % fit_transform方法：先fit再transform，便于一次性完成
        function [X_scaled,obj] = fit_transform(obj, X)
            obj = obj.fit(X);
            X_scaled = obj.transform(X);
        end
    end
end
