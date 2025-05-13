function [X,FSID,X2] = FeatureSelection(X,W,k,varargin)
    % Initialize X2 and flag to check if X2 is provided
    X2 = {};
    flag = 0;
    % Parse optional arguments
    for i = 1:2:length(varargin)
        switch varargin{i}
            case 'X2'
                X2 = varargin{i+1};
                flag = 1;
        end
    end

    % Get the number of datasets
    m = numel(X);
    % Initialize a vector to store the number of features in each dataset
    d_ = zeros(m,1);
    % Calculate the number of features for each dataset
    for i=1:m
        d_(i) = size(X{i},2);
    end

    % Check if W is not a cell array (i.e., a single weight matrix)
    if ~iscell(W)
        % If X is a cell array, concatenate all datasets into a single matrix
        if iscell(X)
            X_ = [];
            X2_ = [];
            for m=1:numel(X)
                X_ = [X_,X{m}];
                if flag
                    X2_ = [X2_,X2{m}];
                end
            end
            X = {X_};
            X2 = {X2_};
            W = {W};
        end
        % Update the number of features to the total number of features in the concatenated matrix
        d_ = size(X_,2);
    end
    % Get the maximum number of features across all datasets
    d = max(d_);
    % Initialize the feature selection ID matrix with NaN values
    FSID = NaN(m,d);
    % Iterate over each dataset and perform feature selection
    for ii = 1:numel(W)
        % Calculate the L2 norm of each column in the weight matrix
        diag_values = vecnorm(W{ii},2,2);
        % Sort the columns based on their norms in descending order
        [~, sorted_indices] = sort(diag_values,'descend','ComparisonMethod','abs');
        % Select the top k features
        top10_indices = sorted_indices(1:min(k, d_(ii)));
        % Select the corresponding features from the dataset
        X{ii} = X{ii}(:,top10_indices);
        % Store the indices of the selected features
        FSID(ii,1:numel(top10_indices)) = top10_indices;
        % If X2 is provided, select the corresponding features from the secondary dataset
        if flag
            X2{ii} = X2{ii}(:,top10_indices);
        end
    end
end