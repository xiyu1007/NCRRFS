classdef Recorder
    % Recorder class for tracking experimental metrics, results, seeds, and runtime.
    % Used in repeated k-fold cross-validation to store and manage evaluation outputs.

    properties
        % Evaluation metrics
        ACC        % Accuracy
        SEN        % Sensitivity (Recall)
        SPE        % Specificity
        AUC        % Area Under ROC Curve
        F1         % F1-score

        % Training records
        Loss = {}      % Training loss history
        FSID = {}      % Selected feature indices
        nfs = -1;      % Number of selected features

        % Prediction outputs
        labs          % Predicted labels
        decs          % Decision values

        % Experimental parameters
        params        % Fixed parameters used in the experiment

        % Experiment configuration
        r             % Number of repeated experiments
        k             % Number of folds in k-fold cross-validation

        % Random seeds
        evaSeeds      % Seeds used for evaluation
        seeds         % Random seeds for experiment

        runtime = 0   % Total runtime
        cr            % Current repetition index

        pt            % Placeholder or extra information (unused)

        task = NaN;   % Task ID or label

        method = '';  % Method name used

        ins           % Additional instance-specific data
    end

    methods
        function obj = Recorder(r, k, params)
            % Constructor for Recorder class.
            % Initializes properties and preallocates memory for performance.

            if nargin > 0
                obj.cr = 0;        % Initialize current repetition
                obj.r = r;         % Total repetitions
                obj.k = k;         % Folds per repetition

                % Preallocate evaluation metric arrays
                obj.ACC = NaN(r, k);
                obj.SEN = NaN(r, k);
                obj.SPE = NaN(r, k);
                obj.AUC = NaN(r, k);
                obj.F1 = NaN(r, k);

                % Initialize seeds and records
                obj.evaSeeds = [];
                obj.seeds = [];

                obj.Loss = cell(r, k);     % Training loss
                obj.FSID = cell(r, k);     % Feature indices

                obj.labs = cell(r, k);     % Predicted labels
                obj.decs = cell(r, k);     % Decision values

                if nargin > 2
                    obj.params = params;   % Set parameters if provided
                else
                    obj.params = [];       % Empty if not provided
                end
            end
        end

        function obj = log(obj, results)
            % Logs experiment results into the Recorder.
            % Supports logging for one repetition or a cell array of multiple.

            if iscell(results)
                for i = 1:numel(results)
                    result = results{i};
                    obj.cr = obj.cr + 1;
                    cr_ = obj.cr;

                    % Store evaluation metrics
                    obj.ACC(cr_, :) = result.acc;
                    obj.SEN(cr_, :) = result.sen;
                    obj.SPE(cr_, :) = result.spe;
                    obj.AUC(cr_, :) = result.auc;
                    obj.F1(cr_, :) = result.f1;

                    % Store predictions
                    for ck = 1:obj.k
                        obj.labs{cr_, ck} = result.labs{ck};
                        obj.decs{cr_, ck} = result.decs{ck};
                    end
                end
            else
                obj.cr = obj.cr + 1;
                cr_ = obj.cr;

                % Store single result
                obj.ACC(cr_, :) = results.acc;
                obj.SEN(cr_, :) = results.sen;
                obj.SPE(cr_, :) = results.spe;
                obj.AUC(cr_, :) = results.auc;
                obj.F1(cr_, :) = results.f1;
                obj.evaSeeds = results.seeds;

                for ck = 1:obj.k
                    obj.labs{cr_, ck} = results.labs{ck};
                    obj.decs{cr_, ck} = results.decs{ck};
                end
            end
        end

        function [value, stdV] = getMetrics(obj, type)
            % Retrieves mean and standard deviation of a specified metric.

            switch lower(type)
                case 'acc'
                    data = obj.ACC;
                case 'sen'
                    data = obj.SEN;
                case 'spe'
                    data = obj.SPE;
                case 'auc'
                    data = obj.AUC;
                case 'f1'
                    data = obj.F1;
            end

            % Handle NaN values
            if any(isnan(data))
                warning('NaNs found in data; replacing with 0.');
            end
            data(isnan(data)) = 0;

            value = mean(data, 2);
            stdV = std(data, [], 2);
        end

        function save(obj, filename, ~)
            % Saves the Recorder object to a .mat file.
            % Creates directories if they do not exist.

            [parentDir, ~, ~] = fileparts(filename);
            if ~isfolder(parentDir)
                mkdir(fullfile(pwd, parentDir));
            end
            save(filename, 'obj');

            if nargin < 3
                fprintf('Model saved to file: %s\n', filename);
            end
        end

        function obj = load(~, filename)
            % Loads a Recorder object from a .mat file.

            data = load(filename);
            obj = data.obj;
            fprintf('Recorder loaded from: %s\n', filename);
        end

        function [value, stdV] = getM(obj, type)
            % Retrieves overall average and standard deviation for given metric type.

            switch lower(type)
                case 'acc'
                    cleaned_data = obj.ACC(~isnan(obj.ACC));
                case 'sen'
                    cleaned_data = obj.SEN(~isnan(obj.SEN));
                case 'spe'
                    cleaned_data = obj.SPE(~isnan(obj.SPE));
                case 'auc'
                    cleaned_data = obj.AUC(~isnan(obj.AUC));
                case 'f1'
                    cleaned_data = obj.F1(~isnan(obj.F1));
                otherwise
                    error('Invalid type. Valid types: ACC, SEN, SPE, AUC, F1.');
            end

            if isempty(cleaned_data)
                value = 0;
                stdV = 0;
            else
                value = mean(cleaned_data);
                stdV = std(cleaned_data);
            end
        end

        function printMetrics(obj)
            % Prints formatted metrics including mean and standard deviation.

            metrics = {'ACC', 'SEN', 'SPE', 'F1', 'AUC'};

            fprintf('-----------------------------------------\n');
            fprintf('| Item \tMean\tStd \t|\n');
            fprintf('-----------------------------------------\n');

            for i = 1:length(metrics)
                type = metrics{i};
                [value, stdV] = obj.getM(type);
                fprintf('| %-4s \t%5.4f\t%5.4f\t|\n', type, value, stdV);
            end

            fprintf('-----------------------------------------\n');
        end

        function [obj, flag] = update(obj, o1)
            % Updates the Recorder object with a better result if available.
            % Compares based on accuracy, AUC, F1, and number of features.

            [v, ~] = obj.getM('acc');
            [v1, ~] = o1.getM('acc');
            flag = 0;

            if v < v1
                obj = o1;
                flag = 1;
            elseif abs(v - v1) < 1e-5
                [va, ~] = obj.getM('auc');
                [v1a, ~] = o1.getM('auc');
                [vf, ~] = obj.getM('f1');
                [v1f, ~] = o1.getM('f1');

                if va < v1a || ...
                   (abs(va - v1a) < 1e-5 && vf < v1f) || ...
                   (abs(va - v1a) < 1e-5 && abs(vf - v1f) < 1e-5 && o1.nfs < obj.nfs)
                    obj = o1;
                    flag = 1;
                end
            end
        end

        function [labslist, decslist] = plotROC(obj, r, k)
            % Combines labels and decisions from given repetitions/folds and plots ROC.

            if nargin < 3, k = obj.k; end
            if nargin < 2, r = obj.r; end

            labslist = [];
            decslist = [];

            for ridx = 1:r
                for kIdx = 1:k
                    if ~isempty(obj.labs{r, kIdx}) && ~isempty(obj.decs{r, kIdx})
                        labslist = [labslist; obj.labs{r, kIdx}(:)];
                        decslist = [decslist; obj.decs{r, kIdx}(:)];
                    end
                end
            end

            if isempty(decslist) || isempty(labslist)
                fprintf('Error: Either decs or preds is empty.\n');
                return
            end

            [fpr, tpr, ~, ~] = perfcurve(labslist(:), decslist(:), 1);

            figure;
            colors = ColorMap(2);
            plot(fpr, tpr, 'Color', colors(1, :), 'LineWidth', 1);
            title(['ROC Curve for Experiment ' num2str(r)]);
            xlabel('False Positive Rate (FPR)');
            ylabel('True Positive Rate (TPR)');
        end

        function loss = plotLoss(obj, be, r, k, color)
            % Plots loss curves from recorded training history.

            if nargin < 2, be = 1; end
            if nargin < 3, r = 1; end
            if nargin < 4, k = 0; end
            if nargin < 5, color = ColorMap(10); end

            if k == 0
                loss = obj.Loss(r, :);
            else
                loss = obj.Loss(r, k);
            end

            figure;
            hold on;

            for i = 1:numel(loss)
                L = loss{i};
                L = L(:, be:end);
                len = size(L, 2);

                if iscell(color)
                    c = color{i};
                elseif r ~= 0 && isscalar(color)
                    c = color;
                else
                    c = color(i, :);
                end

                plot(1:len, L, 'Color', c, ...
                    'Marker', 'o', 'MarkerSize', 3, ...
                    'LineWidth', 1, 'DisplayName', ['Loss ' num2str(i)]);
            end

            hold off;
            legend show;
            xlabel('Epoch');
            ylabel('Loss');
            title('Loss');
        end
    end
end
