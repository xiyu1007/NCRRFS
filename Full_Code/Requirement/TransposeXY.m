function [X,Y] = TransposeXY(X,Y)
    if nargin < 2
        Y = [];
    end
    Y = Y';
    for m=1:numel(X)
        X{m} = X{m}';
    end
end

