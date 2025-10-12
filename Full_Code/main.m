clc;
close all;
clear;

%% ------------------------------------------------------------------------
%  Add Required Paths
% -------------------------------------------------------------------------
addpath(genpath('Method_Requirement'));
addpath(genpath('Method'));
addpath(genpath('Requirement'));

%% ------------------------------------------------------------------------
%  Data Preparation
% -------------------------------------------------------------------------
DataPath = {
    'Datasets\ADNI2\DATA_MRI.csv';
    'Datasets\ADNI2\DATA_PET.csv'
};

Group = {'AD', 'CN'};           % Target groups: AD = Alzheimer's Disease, CN = Control
pre_path = 'ADNI2\AC';           % Subdirectory for saving results

% Load ADNI dataset
%   X: 1×M cell array, each cell is an n×d_m feature matrix
%   Y: n×c one-hot label matrix
[X, Y] = getADData(DataPath, Group, 15, false);

% Get dataset info
[n, c, M, d] = getDataInfo(X, Y);

task = 'AC';                    % Task name (AD vs CN classification)

%% ------------------------------------------------------------------------
%  Experiment Configuration
% -------------------------------------------------------------------------
cp = 1;                         % Skip previously completed parameters if true
save_results = 1;               % Whether to save results to disk
parrun = 1;                     % Parallel run flag (1 = parallel, 0 = sequential)
dep = [];                       % Default parameters (used if recorder is not empty)

% Select method and initialize parameters
method = 'NCRRFS';
params = NCRRFS().init_param([10,0.1,10,100,0.7,10]);
dep = [10,0.1,10,100,0.7,10];    % Default parameters
np = 6;                         % Number of parameters

%% ------------------------------------------------------------------------
%  Load Existing Results (if available)
% -------------------------------------------------------------------------
outpath = 'output\';
result_file = fullfile(outpath, pre_path, [method, '_re.mat']);

if exist(result_file, 'file') && save_results
    recorder = load(result_file).obj;
else
    recorder = [];
end

%% ------------------------------------------------------------------------
%  Run LRTLS
% -------------------------------------------------------------------------
% run2() can be used for parameter grid search (varying two parameters at a time)
% Parameters:
%   np       : number of parameters
%   dep      : default parameter values (used if recorder is empty)
%   parrun   : parallel execution flag
%   cp       : skip completed parameters
%   r, k     : number of repetitions and cross-validation folds (default 10)
%   recorder : stores previous experiment results
%
% Example (uncomment to use):
% recorder = run2(np, method, X, Y, recorder, cp, parrun, task, pre_path, outpath, dep);

%% ------------------------------------------------------------------------
%  Run with Fixed Parameters
% -------------------------------------------------------------------------
% seeds: random seeds for repeated experiments to ensure reproducibility
seeds = [4224 9927 283 7120 1486 5491 9666 5540 3654 5167];
r = 10;     % number of repetitions
k = 10;     % number of folds (cross-validation)

% Run the model
s_time = tic;
recorder = run(method, X, Y, params, r, k, ...
    's', seeds, ...
    'cp', cp, ...
    'recorder', recorder, ...
    'pr', parrun, ...
    'task', task);

%% ------------------------------------------------------------------------
%  Result Analysis
% -------------------------------------------------------------------------
% Print overall metrics
recorder.printMetrics();

% Plot ROC and loss curves for the first view
recorder.plotROC(1);
recorder.plotLoss(1);

% Display accuracy and standard deviation
elapsed_time = toc(s_time);
[acc, stdAcc] = recorder.getM('acc');

fprintf('Runtime: %.2fs\t %s \tAcc = %.4f (±%.4f)\n', ...
    elapsed_time, mat2str([recorder.nfs, recorder.params]), acc, stdAcc);

% Save the recorder object
if save_results
    recorder.save(result_file);
end
