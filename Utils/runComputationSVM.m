function recorder = runComputationSVM(np,method,X,Y,recorder,cp,parrun,task,pre_path,outpath,defaultP)
    %%
    seeds = [4224 9927 283 7120 1486 5491 9666 5540 3654 5167];
    r = 10;
    k = 10;
    % seeds = randi([1, 10000], 1, r);
    if ~exist('defaultP','var') || isempty(defaultP)
        defaultP = [1 1 1, 1 1 1,1,1];
    end
    fix = [];
    %%
    combos = paramSet2(np,fix);
    % combos = combos(end:-1:1, :); 
    for i=1:size(combos,1)
        fix2param = combos(i,:);
        fixed_idx = ~isinf(fix2param);  
        if isempty(recorder)
            if ~isempty(defaultP)
                fix2param(fixed_idx) = defaultP(fixed_idx);
            else
                fix2param(fixed_idx) = 1;
            end
        else
            fix2param(fixed_idx) = recorder.params(fixed_idx);  % 用param中对应值替换
        end
        params = feval(method).init_param(fix2param);
        fprintf('Length of params: %d\n',size(params,1));

        %%
        s_time = tic;
        recorder = run(method,X,Y,params,r,k,'s',seeds,'cp',cp,'rp',1,'recorder',recorder,'pr',parrun,'task',task);
        elapsed_time = toc(s_time);
        [acc,stdAcc] = recorder.getM('acc');
        fprintf('runtime: %.2fs.   %s \tacc=%.4f(±%.4f)\n', elapsed_time,mat2str([recorder.params]),acc,stdAcc);
        %%
        recorder.save([outpath,pre_path,'\',method,'_re.mat'])
    end
end
