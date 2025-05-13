function [x_train, x_test, y_train, y_test] = cv_split(X, Y, cv,z_score)
    % cv_split 根据交叉验证为每个模态数据分配训练集和测试集，并返回对应的标签
    %
    % 输入：
    %   X  - 一个 cell 数组，每个元素是一个模态的数据矩阵 (N x D)
    %   Y  - 一个标签向量 (N x 1)，对应数据 X
    %   cv - 一个交叉验证对象（cvpartition），用于分割数据
    %
    % 输出：
    %   train_data   - 一个 cell 数组，包含 K 个折，每个折是一个 cell，其中包含每个模态的训练数据（N_train x D）
    %   test_data    - 一个 cell 数组，包含 K 个折，每个折是一个 cell，其中包含每个模态的测试数据（N_test x D）
    %   train_labels - 一个 cell 数组，包含 K 个折，每个折是训练标签（N_train x 1）
    %   test_labels  - 一个 cell 数组，包含 K 个折，每个折是测试标签（N_test x 1）
    if ~exist("z_score",'var'), z_score = 0; end

    num_modalities = numel(X);  % 模态的数量
    num_folds = cv.NumTestSets; % 交叉验证的折数

    % 初始化输出
    x_train = cell(num_folds, 1);
    x_test = cell(num_folds, 1);
    y_train = cell(num_folds, 1);
    y_test = cell(num_folds, 1);

    % 对每个折进行处理
    for fold = 1:num_folds
        % 获取当前折的训练集和测试集的索引
        train_index = cv.training(fold);
        test_index = cv.test(fold);
        
        % 分别为训练数据和测试数据初始化每个模态的 cell 数组
        x_train{fold} = cell(num_modalities, 1);
        x_test{fold} = cell(num_modalities, 1);
        
        % 初始化训练和测试标签
        y_train{fold} = Y(train_index,:);
        y_test{fold} = Y(test_index,:);
        
        % 为每个模态的数据提取对应的训练和测试数据
        for modality = 1:num_modalities
            % 训练数据
            xtrain = X{modality}(train_index, :);
            xtest = X{modality}(test_index, :);
            if z_score
                scaler = Scaler();
                [xtrain,scaler,] = scaler.fit_transform(xtrain);
                xtest = scaler.transform(xtest);
            end

            x_train{fold}{modality} = xtrain;
            % 测试数据
            x_test{fold}{modality} = xtest;
        end
    end
end
