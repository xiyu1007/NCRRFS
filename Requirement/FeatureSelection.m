function [X,FSID,X2] = FeatureSelection(X,W,k,varargin)
    X2 = {};
    flag = 0;
    for i = 1:2:length(varargin)
        switch varargin{i}
            case 'X2'
                X2 = varargin{i+1};
                flag = 1;
        end
    end

    m = numel(X);
    d_ = zeros(m,1);
    for i=1:m
        d_(i) = size(X{i},2);
    end

    if ~iscell(W)
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
        % k = numel(X) * k;
        d_ = size(X_,2);
    end
    d = max(d_);
    FSID = NaN(m,d);
    for ii = 1:numel(W)
        diag_values = vecnorm(W{ii},2,2);
        [~, sorted_indices] = sort(diag_values,'descend','ComparisonMethod','abs');
        top10_indices = sorted_indices(1:min(k, d_(ii)));
        X{ii} = X{ii}(:,top10_indices);
        FSID(ii,1:numel(top10_indices)) = top10_indices;
        if flag
            X2{ii} = X2{ii}(:,top10_indices);
        end
    end
end

