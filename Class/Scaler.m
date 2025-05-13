classdef Scaler
    % Scaler class for feature-wise standardization of data matrix X.
    % Supports flexible normalization along rows or columns.

    properties
        meanX      % Stores the mean of the training data
        stdX       % Stores the standard deviation of the training data
        epsilon = 1e-8  % Small constant to avoid division by zero
        dim = 1;   % Normalization dimension (default is 1, i.e., column-wise for n*d)
    end

    methods
        function obj = Scaler(dim)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Constructor: Scaler
            %
            % Description:
            %   Initializes the Scaler object. Optionally allows specifying
            %   the dimension for normalization (1 for column-wise, 2 for row-wise).
            %
            % Input:
            %   - dim (optional): normalization dimension (default = 1)
            %
            % Output:
            %   - obj : an instance of the Scaler class
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if nargin > 0
                obj.dim = dim;
            end
        end

        function obj = fit(obj, X)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Method: fit
            %
            % Description:
            %   Computes and stores the mean and standard deviation of the
            %   input data along the specified dimension.
            %
            % Input:
            %   - X : input data matrix (n*d)
            %
            % Output:
            %   - obj : updated Scaler object with stored mean and std
            %
            % Notes:
            %   Adds no epsilon here; division safety is handled in transform.
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            obj.meanX = mean(X, obj.dim);            % Mean along dim
            obj.stdX = std(X, 0, obj.dim);           % Unbiased std deviation
        end

        function X_scaled = transform(obj, X)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Method: transform
            %
            % Description:
            %   Applies standardization to the input data using the stored
            %   mean and standard deviation from the fit method.
            %
            % Input:
            %   - X : input data matrix to be scaled
            %
            % Output:
            %   - X_scaled : standardized data matrix
            %
            % Notes:
            %   Adds epsilon to the denominator to prevent divide-by-zero.
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            X_scaled = (X - obj.meanX) ./ (obj.stdX + obj.epsilon);
        end

        function [X_scaled, obj] = fit_transform(obj, X)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Method: fit_transform
            %
            % Description:
            %   Combines fit and transform operations for convenience.
            %   First computes the normalization statistics, then applies
            %   them to standardize the data.
            %
            % Input:
            %   - X : input data matrix
            %
            % Output:
            %   - X_scaled : standardized data matrix
            %   - obj      : updated Scaler object
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            obj = obj.fit(X);
            X_scaled = obj.transform(X);
        end
    end
end
