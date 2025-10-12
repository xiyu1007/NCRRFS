function [X,Y] = shuffledData(X,Y,seeds)
    if nargin > 2
        rng(seeds);
        shuffled_indices = randperm(size(Y,1)); % 生成随机索引
        rng('shuffle');
    else
        shuffled_indices = randperm(size(Y,1)); % 生成随机索引
    end
    % X: n*d
    Y = Y(shuffled_indices,:); 
    for ii=1:length(X)
        X{ii} = X{ii}(shuffled_indices,:);
    end
end

