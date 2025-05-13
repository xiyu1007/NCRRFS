function model = mk_svm(X, Y,options)
    % mk_svm - 用于训练 SVM 模型的函数
    % 输入:
    %   X - 训练特征矩阵 (每一行是一个样本的特征)
    %   y - 训练标签向量 (每个样本的标签)
    % 输出:
    %   model - 训练好的 SVM 模型
    
    % 检查输入数据是否正确
    if nargin < 3
        options = '-t 4 -q';  % -t 4 表示使用预计算核函数，-q 表示静默模式
    end
    
    % 训练 SVM 模型
    N = size(Y,1);
    X_ = [(1:N)', X];

    model = svmtrain(Y, X_, options); %#ok<SVMTRAIN> % '-t 4' 表示预计算核
end
