clc;
close all;
clear;

%% A demo on how to run

addpath(genpath('Requirement'))
addpath('Method')
addpath(genpath('Method_Requirement'))

%% Generate data - running example
c = 2;      % Number of classes
n = 100;    % Number of samples
d = 50;     % Feature dimension
M = 3;      % Number of views or feature sets

%% Generate Y (one-hot encoding)
Y = zeros(n, c);
labels = randi([1 c], n, 1);   % Randomly generate class labels
for i = 1:n
    Y(i, labels(i)) = 1;
end

%% Generate X (M cells, each one an n×d matrix)
X = cell(1, M);
for m = 1:M
    X{m} = rand(n, d);  % Randomly generate feature matrices
end

%% Initialization
method = 'NCRRFS';
params = NCRRFS().init_param([10,0.1,10,10,0.7,10]);
ins = feval(method);
ins.verbose = 1;
param = params(1, :);

%% Run 
% LRTLS has 6 parameters. 
% The optimal parameters for each task can be found in the output .mat file.
%
% Input:
%   X: 1×M cell array, each cell is an n×d_m matrix
%   Y: n×c one-hot label matrix
%
% Output:
%   LRTLS object. You can access the feature selection matrices through ins.W
%   (a 1×M cell array). Each matrix is of size d×(5c), where c is the number of classes.
ins = ins.run(X, Y, param, 42);
