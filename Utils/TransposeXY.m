function [X,Y] = TransposeXY(X,Y)
    % Check if Y is not provided and initialize it as an empty matrix if necessary
    if nargin < 2
        Y = [];
    end
    % Transpose Y if it is not empty
    if ~isempty(Y)
        Y = Y';
    end
    % Transpose each matrix in the cell array X
    for m=1:numel(X)
        X{m} = X{m}';
    end
end