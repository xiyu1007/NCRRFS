clc;
clear;
close all;

DataPath = {
    'Datasets\ADNI2\DATA_MRI.csv';
    'Datasets\ADNI2\DATA_PET.csv'
};

%%
seeds = [4224 9927 283 7120 1486 5491 9666 5540 3654 5167];
r = 10;
k = 10;

Group = {'AD','CN'};
pre_path = 'ADNI2\AC';

[X,Y] = getADData(DataPath,Group,14,false); 

[n, c, M, d] = getDataInfo(X,Y);
%%
method = 'MKSVM';
%%

parts = strsplit(pre_path, '\'); % 按 '_' 分割字符串
task = parts{2}; % 提取 '_' 前面的部分

re = svmrun(method, X, Y, seeds, task, r, k,0);
re.printMetrics();
re.plotROC(1);
disp('----------------------')


re.save(['output\',pre_path,'\',method,'_re.mat'])

%%
function recorder = svmrun(method, X, Y, seeds, task, numRepeat, numFold,useParfor)
    cols = cellfun(@(x) size(x,2), X);
    FeatureSpace = min(cols);

    taskName = task;
    numFeatures = numel(FeatureSpace);
    % numViews = numel(X);
    numViews = numel(X);
    numTotal = numRepeat*numFold;
    Xtrain = cell(1,numTotal);
    Xtest = cell(1,numTotal);
    Ytrain = cell(1,numTotal);
    Ytest = cell(1,numTotal);

    for i = 1:numRepeat
        [Xd, Yd] = shuffledData(X, Y, seeds(i));
        rng(seeds(i));
        cv = cvpartition(Yd(:,1), 'KFold', numFold, 'Stratify', true);
        bi = (i-1)*numFold + 1;
        ei = bi + numFold - 1;
        [Xtrain(1,bi:ei), Xtest(1,bi:ei),Ytrain(1,bi:ei), Ytest(1,bi:ei)] = cv_split(Xd, Yd, cv);
        % cvList{i} = cv;
    end
    tic;

    tempResults = cell(1, numFeatures);

    if useParfor
        parfor idx = 1:numTotal
            tempResults{idx} = Evaluate(Xtrain{idx}, Xtest{idx}, Ytrain{idx}, Ytest{idx}, 'kernel', 'linear','FeatureSpace',FeatureSpace);
        end
    else
        for idx = 1:numTotal
            tempResults{idx} = Evaluate(Xtrain{idx}, Xtest{idx}, Ytrain{idx}, Ytest{idx}, 'kernel', 'linear','FeatureSpace',FeatureSpace);
        end
    end

        resultsList = unFlodResult(tempResults,numRepeat,numFold,numFeatures);
        recorder = Recorder(numRepeat, numFold).log(resultsList(1,:));

        [acc, stdAcc] = recorder.getM('acc');
        elapsed = toc;
        if 1
            fprintf('%-.2fs | \t\t\t acc=%-5.4f(±%-5.4f)\n',elapsed, acc, stdAcc);
        end

        recorder.seeds = seeds;
        recorder.method = method;
        recorder.task = taskName;
        recorder.ins = numViews;
        % recorder.save(['output\',taskName,'_',method,'_re_temp.mat'], 0);

end


function resultsList = unFlodResult(resultCache,numRepeat,numFold,numFeatures)
        numTotal = numel(resultCache);
        tempList.acc = NaN(numFeatures,numTotal);
        tempList.sen = NaN(numFeatures,numTotal);
        tempList.spe = NaN(numFeatures,numTotal);
        tempList.f1 = NaN(numFeatures,numTotal);
        tempList.auc = NaN(numFeatures,numTotal);
        tempList.labs = cell(numFeatures,numTotal);
        tempList.decs = cell(numFeatures,numTotal);

    for i=1:numel(resultCache)
        tempList.acc(:,i) = resultCache{i}.acc;
        tempList.sen(:,i) = resultCache{i}.sen;
        tempList.spe(:,i) = resultCache{i}.spe;
        tempList.f1(:,i) = resultCache{i}.f1;
        tempList.auc(:,i) = resultCache{i}.auc;
        tempList.labs(:,i) = resultCache{i}.labs;
        tempList.decs(:,i) = resultCache{i}.decs;
    end

    resultsList = cell(numFeatures,numRepeat);
    for i = 1:numFeatures
        for j=1:numRepeat
            bj = (j-1)*numFold+1;
            ej = bj + numFold - 1;
            temp.acc = tempList.acc(i,bj:ej);
            temp.sen = tempList.sen(i,bj:ej);
            temp.spe = tempList.spe(i,bj:ej);
            temp.f1 =  tempList.f1(i,bj:ej);
            temp.auc = tempList.auc(i,bj:ej);
            temp.labs = tempList.labs(i,bj:ej);
            temp.decs = tempList.decs(i,bj:ej);
            resultsList{i,j} = temp;
        end
    end
end