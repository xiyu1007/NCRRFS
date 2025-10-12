clc;
clear;
close all;

addpath(genpath('Method'));
addpath(genpath('Requirement'));

method = 'My';
path = ['output\ADNI2\AC\',method,'_re.mat'];

recorder = load(path).obj;

recorder.printMetrics()
recorder.plotLoss(1); % 第一次重复的k折实验的所有loss

nfs = recorder.nfs;
pt = recorder.pt;
best_param = recorder.params;


% leg = {'𝛼', '𝛽', '𝛾', '𝜆','h','all'};
% pt.plotAblation([],leg) 

eparam = best_param;
eparam(2) = -1; % 绘制第二个参数随特征变化
[p,m] = pt.plotEffect([nfs*2,eparam], 1, 1);

barparam = best_param;
barparam([1,2]) = -1; % 绘制第1,2个参数组合变化
pName = {'\alpha', '\beta', 'acc'};
pt.plotBar([nfs*2,barparam],pName, 1);

