function recorder = run2(np,method,X,Y,recorder,cp,parrun,task,pre_path,outpath,defaultP)
    %%
    % verbose = 0;
    seeds = [4224 9927 283 7120 1486 5491 9666 5540 3654 5167];

    r = 10;
    k = 10;
    % seeds = randi([1, 10000], 1, r);
    if ~exist('defaultP','var') || isempty(defaultP)
        defaultP = [1 1 1, 1 1 1,1,1];
    end
    % defaultP = [1 1 1, 1 1 1];
    fix = [];
    %%

    combos = paramSet2(np,fix);
    combos = combos(end:-1:1, :); % 从最后一列到第一列反向索引

    for i=1:size(combos,1)
        fix2param = combos(i,:);
        % 找出不为Inf的位置，即两个是Inf的参数，这一轮调整这两个参数，其余参数被设置为默认参数
        fixed_idx = ~isinf(fix2param);   
        % 下面是设置默认参数
        if isempty(recorder) % 如果有recorder从其中加载默认参数
            if ~isempty(defaultP)
                fix2param(fixed_idx) = defaultP(fixed_idx);
            else
                fix2param(fixed_idx) = 1; % 如果没有将所有参数默认为1.
            end
        else
            fix2param(fixed_idx) = recorder.params(fixed_idx);  % 用param中对应值替换
        end
        % 这里将需要调整的两个参数利用方法的init_param初始化
        params = feval(method).init_param(fix2param);
        % params = unique(params,"rows");
        fprintf('Length of params: %d\n',size(params,1));

        %%
        s_time = tic;
        recorder = run(method,X,Y,params,r,k,'s',seeds,'cp',cp,'rp',1,'recorder',recorder,'pr',parrun,'task',task);
        elapsed_time = toc(s_time);
        [acc,stdAcc] = recorder.getM('acc');
        fprintf('runtime: %.2fs.   %s \tacc=%.4f(±%.4f)\n', elapsed_time,mat2str([recorder.params]),acc,stdAcc);
        %%

        recorder.save([outpath,pre_path,'\',method,'_re.mat'])
        if strcmp(recorder.method,'My')
            recorder.save(['AL_output\',pre_path,'\',method,'_re.mat'])
        end
    end
end
