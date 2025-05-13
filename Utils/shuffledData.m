function [X,Y] = shuffledData(X,Y,seeds)
    % Check if a seed is provided
    if nargin > 2
        % Set the random number generator seed for reproducibility
        rng(seeds);
        % Generate a random permutation of indices based on the number of rows in Y
        shuffled_indices = randperm(size(Y,1)); % Generate random indices
        % Reset the random number generator to its default state
        rng('shuffle');
    else
        % Generate a random permutation of indices without a specific seed
        shuffled_indices = randperm(size(Y,1)); % Generate random indices
    end
    % Shuffle Y based on the generated indices
    Y = Y(shuffled_indices,:);
    % Shuffle each matrix in the cell array X based on the same indices
    for ii=1:length(X)
        X{ii} = X{ii}(shuffled_indices,:);
    end
end