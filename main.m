%%
    % % % for svmtrain ：https://www.csie.ntu.edu.tw/~cjlin/libsvm/
    % % % for ADNI datasets：https://adni.loni.usc.edu/
    % % 
    % % %==========================================================================
    % % % Script:     main.m
    % % % Purpose:    Conduct SVM-based classification experiments on the ADNI dataset.
    % % %             This includes data loading, preprocessing, parameter tuning,
    % % %             and evaluation (metrics, ROC, loss).
    % % %
    % % % Usage:      Customize the 'DataPath', 'Group', and 'pre_path' variables,
    % % %             then run the script to perform classification.
    % % %
    % % % Requires:   - LIBSVM (https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
    % % %             - ADNI dataset (https://adni.loni.usc.edu/)
    % % %             - Custom SVM wrapper and evaluation utilities (`My.m`, `run.m`, etc.)
    % % %
    % % % Dataset:    ADNI1 (Alzheimer's Disease Neuroimaging Initiative)
    % % %             Includes features from MRI and PET imaging.
    % % %
    % % % Author:     Xi Guo
    % % % Created:    None
    % % %==========================================================================
    % % 
    % % clc;
    % % clear;
    % % close all;
    % % 
    % % %==========================================================================
    % % % DATA CONFIGURATION
    % % %==========================================================================
    % % % Define file paths for feature CSVs (MRI, PET data)
    % % DataPath = {
    % %     'Datasets\ADNI1\DATA_MRI.csv';
    % %     'Datasets\ADNI1\DATA_PET.csv'
    % % };
    % % 
    % % % Define classification groups: AD vs CN (Alzheimer's vs Cognitively Normal)
    % % Group = {'AD','CN'};
    % % 
    % % % Define experiment folder path (e.g., for saving results)
    % % pre_path = 'ADNI1\AC';
    % % 
    % % % Load feature matrix X and label vector Y
    % % % getADData loads and merges multiple data files for selected groups
    % % [X,Y] = getADData(DataPath, Group, 14, false); 
    % % 
    % % % Get dataset information: sample count, class count, feature count, etc.
    % % [n, c, M, d] = getDataInfo(X, Y);
    % % 
    % % % Extract task name (e.g., 'AC') from path
    % % parts = strsplit(pre_path, '\');
    % % task = parts{2};
    % % 
    % % %==========================================================================
    % % % CONTROL FLAGS
    % % %==========================================================================
    % % cp = 0;         % Whether to use cached predictions
    % % save = 1;       % Whether to save results
    % % if cp == 0
    % %     save = 0;
    % % end
    % % parrun = 1;     % Whether to enable parallel execution
    % % 
    % % %==========================================================================
    % % % EXPERIMENT METHOD CONFIGURATION
    % % %==========================================================================
    % % method = 'My'; % Use custom SVM implementation (defined in My.m)
    % % 
    % % % Initialize SVM parameters
    % % params = My().init_param([100, 0.01, 100, 1, 0.01, 0.5]); % ADNI1 AC Task best parameters
    % % np = 6; % Number of parameters
    % % 
    % % %==========================================================================
    % % % OUTPUT & CACHE SETUP
    % % %==========================================================================
    % % outpath = 'output\';
    % % 
    % % % Check for cached result
    % % if exist([outpath, pre_path, '\', method, '_re.mat'], 'file') && save
    % %     recorder = load([outpath, pre_path, '\', method, '_re.mat']).obj;
    % % else 
    % %     recorder = [];
    % % end
    % % 
    % % %==========================================================================
    % % % MAIN RUN LOGIC (choose one of the following options)
    % % %==========================================================================
    % % s_time = tic;
    % % 
    % % %--------------------------
    % % % Option 1: Parameter sweep with combinations (commented by default)
    % % %--------------------------
    % % % recorder = runComputationSVM(np, method, X, Y, recorder, cp, parrun, task, pre_path, outpath);
    % % 
    % % %--------------------------
    % % % Option 2: Run with fixed parameters (10-fold, 10 times)
    % % %--------------------------
    % % % Fixed random seeds (for reproducibility)
    % % % seeds = [4224 9927 283 7120 1486 5491 9666 5540 3654 5167]; 
    % % % r = 10; % repetitions
    % % % k = 10; % k-fold
    % % % recorder = run(method, X, Y, params, r, k, 's', seeds, 'cp', cp, ...
    % % %                'rp', 1, 'recorder', recorder, 'pr', parrun, 'task', task);
    % % 
    % % %==========================================================================
    % % % EVALUATION AND OUTPUT
    % % %==========================================================================
    % % recorder.printMetrics();   % Print overall evaluation metrics
    % % recorder.plotROC(1);       % Plot ROC curve for class 1 (typically positive class)
    % % recorder.plotLoss(1);      % Plot training loss over iterations
    % % 
    % % elapsed_time = toc(s_time);
    % % 
    % % % Print runtime and best accuracy summary
    % % [acc, stdAcc] = recorder.getM('acc');
    % % fprintf('runtime: %.2fs.   %s \tacc=%.4f(±%.4f)\n', ...
    % %     elapsed_time, mat2str([recorder.nfs, recorder.params]), acc, stdAcc);
    % % 
    % % % Save results if enabled
    % % if save
    % %     recorder.save([outpath, pre_path, '\', method, '_re.mat']);
    % % end

%%
% An example
%==========================================================================
% OPTIONAL: LOAD SAVED RESULT FOR REVIEW
%==========================================================================

clc;
clear;
close all;
recorder = load('output\ADNI1\AC\My_re.mat').obj;
recorder.printMetrics();
recorder.plotROC(2);
recorder.plotLoss(1);

maxparam = [-1, recorder.params];
pt = recorder.pt;
[p,m] = pt.plotEffect(maxparam); % num of feature and acc

