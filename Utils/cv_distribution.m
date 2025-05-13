function cv_distribution(cv,Y,k)
    if nargin < 3, k=1; end
    fprintf('-----------------------------\n');
    counts = tabulate(Y);
    disp('Current Class Distribution:');

    for i = 1:size(counts, 1)
        fprintf('%g\t%g\t%.2f%%\n', counts(i, 1), counts(i, 2), counts(i, 3));
    end
    if k==-1
        for k = 1:cv.NumTestSets
            fprintf('-----------------------------\n');
            % 获取训练集和测试集的索引
            trainIdx = training(cv, k);
            testIdx = test(cv, k);
            % 输出每一折训练集和测试集的类别分布
            fprintf('Train-Fold %d:\n', k);
            counts = tabulate(Y(trainIdx));
            fprintf('Value\tCount\tPercent\n');
            for i = 1:size(counts, 1)
                fprintf('%g\t%g\t%.2f%%\n', counts(i, 1), counts(i, 2), counts(i, 3));
            end
            fprintf('\nTest-Fold %d:\n', k);
            counts = tabulate(Y(testIdx));
            fprintf('Value\tCount\tPercent\n');
            for i = 1:size(counts, 1)
                fprintf('%g\t%g\t%.2f%%\n', counts(i, 1), counts(i, 2), counts(i, 3));
            end
            fprintf('-----------------------------\n');
        end
    end
end

