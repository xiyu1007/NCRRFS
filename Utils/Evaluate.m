function results = Evaluate(X_train,X_test,Y_train,Y_test, varargin)
    % profile on
    % Set default values
    % seeds = 0;
    kernel = 'linear'; 
    % kernel = {'rbf'}; 
    FeatureSpace = 1:1:10;
    % opt_weight = [0.5,0.5];
    opt_weight = ones(1, numel(X_train)) ./ numel(X_train);
    % Parse optional arguments
    for i = 1:2:length(varargin)
        switch varargin{i}
            case 'kernel'
                kernel = varargin{i+1};
            case 'FeatureSpace'
                FeatureSpace = varargin{i+1};
            otherwise
                error('Unknown parameter: %s', varargin{i});
        end
    end
    Y_train = standardizeLabels(Y_train);
    Y_test = standardizeLabels(Y_test);
    numFeatures = numel(FeatureSpace);
    results.acc = NaN(numFeatures,1);
    results.sen = NaN(numFeatures,1);
    results.spe = NaN(numFeatures,1);
    results.f1 = NaN(numFeatures,1);
    results.auc = NaN(numFeatures,1);
    results.labs = cell(numFeatures,1);
    results.decs = cell(numFeatures,1);

    for i = 1:numFeatures
        nf = FeatureSpace(i);
        [nfxtrain,nfxtest] = selectX(X_train,X_test,nf);

        k_train = mk_kernel(nfxtrain,'weights',opt_weight,'kernels',{kernel});
        model = mk_svm(k_train, Y_train);
        k_test = mk_kernel(nfxtest,'X2',nfxtrain,'weights',opt_weight,'kernels',{kernel});
        [preds, ~, decs]= mk_svm_predict(model,k_test,Y_test); % col v

        [acc, sen, spe, f1, auc] = calculate_metrics(Y_test, preds, decs);
        results.acc(i,1) = acc;
        results.sen(i,1) = sen;
        results.spe(i,1) = spe;
        results.f1(i,1) = f1;
        results.auc(i,1) = auc;
        results.labs{i,1} = Y_test;
        results.decs{i,1} = decs;
    end

end

function [X1,X2] = selectX(X1,X2,nf)
    for m=1:numel(X1)
        X1{m} = X1{m}(:,1:nf);
        X2{m} = X2{m}(:,1:nf);
    end
end


%% 辅助函数：统一标签格式为 [-1,1]
function Y_out = standardizeLabels(Y_in)
    % 如果全部已经为 -1 与 1 则保持，否则转换第一列数据：假设原始标签为0/1
    if ~all(ismember(Y_in, [-1, 1]))
        Y_out = Y_in(:,1) * 2 - 1;
    else
        Y_out = Y_in(:,1);
    end
end

% 辅助函数，用于计算各项评估指标
function [acc, sen, spe, f1, auc] = calculate_metrics(y_true, y_pred, decision_values)
    % 计算混淆矩阵
    cm = confusionmat(y_true, y_pred);
    % 提取混淆矩阵中的元素，避免重复索引
    tp = cm(2, 2); % 真阳性
    tn = cm(1, 1); % 真阴性
    fp = cm(1, 2); % 假阳性
    fn = cm(2, 1); % 假阴性
    tol = 1e-8;


    % 计算准确率
    acc = (tp + tn) / (tp + tn + fp + fn + tol);
    % 计算灵敏度（Sensitivity）/召回率（Recall）
    sen = tp / (tp + fn + tol);
    % 计算特异性（Specificity）
    spe = tn / (tn + fp + tol);
    % 计算精确率（Precision）
    precision = tp / (tp + fp + tol);
    % 计算F1分数
    f1 = 2 * (precision * sen) / (precision + sen + tol);
    % 计算AUC
    [~, ~, ~, auc] = perfcurve(y_true, decision_values, 1);
end