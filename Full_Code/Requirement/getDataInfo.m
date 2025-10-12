function [n, c, M, d] = getDataInfo(X, Y)
    M = numel(X);
    [n, c] = size(Y);
    d = zeros(1, M); % 预分配d的大小
    for m = 1:M
        d(m) = size(X{m}, 2);
    end
end