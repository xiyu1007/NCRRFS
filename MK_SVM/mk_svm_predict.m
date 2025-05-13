function [predictions, accuracy, decision_values]= mk_svm_predict(model,X,Y,options)
    if nargin < 4, options = '-q'; end
    N = size(X, 1);  % 测试集样本数
    X_ = [(1:N)', X];  % 添加样本索引列
    [predictions, accuracy, decision_values] = svmpredict(Y, X_, model,options);
end
