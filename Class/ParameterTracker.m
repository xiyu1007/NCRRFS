classdef ParameterTracker
    % ParameterTracker - A class to manage and evaluate sets of parameters along with their performance metrics.
    %
    % This class supports the storage, addition, and analysis of multiple groups of parameters and their
    % corresponding evaluation metrics. It provides various methods to add new parameter sets, find
    % existing ones, visualize parameter influence, and export results to files.
    %
    % Properties:
    %   params      - A matrix where each row represents a set of parameter values.
    %   metrics     - A matrix of evaluation metrics corresponding to each parameter group (acc, sen, spe, f1, auc).
    %   seeds       - A global seed value shared across all parameter groups.
    %   numGroups   - The current number of parameter groups stored.
    %   maxGroups   - The maximum number of parameter groups supported (auto-expandable).
    %   numP        - The number of parameters per group.
    %   method      - (Optional) String or identifier for the method used in the experiments.
    %
    % Methods:
    %   ParameterTracker    - Constructor to initialize the tracker with a parameter count and max size.
    %   addParameters       - Adds a new parameter group and its evaluation metrics.
    %   addMetrics          - Adds or updates metrics for the most recently added parameter group.
    %   findLike            - Returns logical indices of rows in a dataset that match a given parameter.
    %   plotAblation        - Visualizes ablation results by showing the impact of zeroing individual parameters.
    %   plotBar             - Draws a 3D bar plot comparing two variable parameters and a metric.
    %   plotEffect          - Plots how changing one parameter affects a chosen metric.
    %   findParam           - Returns the row index if a parameter group already exists.
    %   checkExist          - Checks whether a parameter group or multiple groups exist in the tracker.
    %   pToFile             - Writes all parameter groups and their metrics to a file.
    %   save                - Saves the entire ParameterTracker instance to a `.mat` file.
    %   load                - Loads a ParameterTracker object from a `.mat` file.
    %
    % Example usage:
    %   tracker = ParameterTracker(5); % Initialize for 5 parameters per group
    %   tracker = tracker.addParameters([0.1, 0.2, 0.3, 0, 0.5], metrics);

    properties
        params      % Matrix storing parameter sets, each row is a set of parameter values
        metrics     % Matrix storing performance metrics for each parameter group [acc, sen, spe, f1, auc]
        seeds       % Global random seed value
        numGroups   % Number of parameter groups currently stored
        maxGroups   % Maximum number of parameter groups the tracker can hold
        numP        % Number of parameters in each parameter group
        method      % Optional identifier or method name (unused)
    end
    
    methods
        function obj = ParameterTracker(NumP, MaxGroups)
            % Constructor to initialize the parameter tracker
            % Input:
            %   NumP      - Number of parameters per group
            %   MaxGroups - (Optional) Maximum number of groups (default = 1000)
            if nargin < 2
                MaxGroups = 1000;
            end
            obj.numP = NumP;
            obj.maxGroups = MaxGroups;
            obj.numGroups = 0;
            obj.params = zeros(MaxGroups, NumP);
            obj.metrics = zeros(MaxGroups, 5); % acc, sen, spe, f1, auc
            obj.seeds = NaN;
        end

       function [obj, acc] = addParameters(obj, param_values, result)
            % Adds a new parameter group and its associated metrics
            % If the group already exists, it is updated.
            % Input:
            %   param_values - 1xN vector of parameter values
            %   result       - Array or cell array of structures containing metric results
            % Output:
            %   acc          - The accuracy (first metric) of the inserted/updated group
            
            if iscell(result)
                mes = zeros(numel(result), 5);
                for r = 1:numel(result)
                    re = result{r};
                    mes(r,:) = [mean(re.acc,'omitnan'), mean(re.sen,'omitnan'), ...
                                mean(re.spe,'omitnan'), mean(re.f1,'omitnan'), ...
                                mean(re.auc,'omitnan')];
                end
                m = mean(mes, 1, 'omitnan');
            else
                m = result;
            end
            acc = m(1);

            idx = findParam(obj, param_values);
            if idx ~= 0
                obj.params(idx, :) = param_values;
                obj.metrics(idx, :) = m;
            else
                if obj.numGroups >= obj.maxGroups
                    obj.maxGroups = obj.maxGroups * 2;
                    obj.params(obj.numGroups + 1:obj.maxGroups, :) = 0;
                    obj.metrics(obj.numGroups + 1:obj.maxGroups, :) = 0;
                end
                obj.numGroups = obj.numGroups + 1;
                obj.params(obj.numGroups, :) = param_values;
                obj.metrics(obj.numGroups, :) = m;
            end
        end
        
        function obj = addMetrics(obj, metrics)
            % Adds or updates metrics for the latest parameter group
            if obj.numGroups == 0
                error('No parameter group found. Add a group first.');
            end
            obj.metrics(obj.numGroups, :) = metrics;
        end

        function rowIndex = findLike(~, ipdata, param)
            % Returns logical row indices where ipdata rows equal param
            rowIndex = ismember(ipdata, param, 'rows');
        end

        function [maxlist,maxparam] = plotAblation(obj,colors,leng,metricIdx)
            % Generates an ablation plot showing impact of zeroed parameters
            % across the selected metric
            % Input:
            %   colors    - Line colors
            %   leng      - Legend entries
            %   metricIdx - Index of metric to plot (default = 1)
            % Output:
            %   maxlist   - Max metric values for each ablation case
            %   maxparam  - Parameter group achieving overall best result
            nump = obj.numP;
            maxlist = zeros(1,nump+1);
            
            if nargin >= 2 && isempty(colors)
                colors = checkColor(colors,nump+10);
            end
            if nargin < 2
                colors = checkColor([],nump+10);
            end

            if nargin < 4, metricIdx=1; end

            data = [obj.params(1:obj.numGroups,:),obj.metrics(1:obj.numGroups,metricIdx)];
            
            figure;
            hold on;
            for ip=2:nump+1
                ipdata = data(data(:,ip) == 0,:);
                if size(ipdata,1) == 0
                    fprintf('continue, no zero to draw in idx: %d\n', ip-1);
                    continue
                end
                ipdata = sortrows(ipdata,'descend');
                [maxlist(ip-1), maxIndex] = max(ipdata(:, end));

                param = ipdata(maxIndex,2:end-1);
                rowIndex = obj.findLike(ipdata(:,2:end-1),param);
                
                ipdata = ipdata(rowIndex,[1, end]);
                ipdata = sortrows(ipdata,'ascend');
                if iscell(colors)
                    c = colors{ip-1};
                else
                    c = colors(ip-1,:);
                end
                if nargin < 3
                    lg = ['no-',num2str(ip-1)];
                else
                    lg = leng{ip-1};
                end
                plot(ipdata(:,1), ipdata(:,2), 'Color', c,'Marker', '.', ...
                    'MarkerSize',1.3 ,'LineWidth', 1.2,'DisplayName', lg);
            end
            
            [maxlist(end), maxIndex] = max(data(:, end));
            param = data(maxIndex,2:end-1);
            nfs = data(maxIndex,1);
            maxparam = [nfs,param];
            id = find(param == 0);
            if ~isempty(id)
                warning('The max value is in paramIdx = %s', mat2str(id));
            else
                if iscell(colors)
                    c = colors{nump+1};
                else
                    c = colors(nump+1,:);
                end
                if nargin >= 3 && numel(leng) >= nump+1
                    lg = leng{nump+1};
                else
                    lg = 'all';
                end
                rowIndex = obj.findLike(data(:,2:end-1),param);
                data = data(rowIndex,[1, end]);
                data = sortrows(data,'ascend');
                plot(data(:,1), data(:,2), 'Color', c,'Marker', '.', ...
                'MarkerSize',1.3 ,'LineWidth', 1.2,'DisplayName', lg);
            end
            legend('show');
            legend('Location', 'best');
            
        end
        
        function plotBar(obj, params,pName, metricIdx)
            % Plots a 3D bar chart for two varying parameters and one selected metric
            % Input:
            %   params    - Vector containing fixed values or -1 for variable parameters
            %   pName     - Cell array of axis labels
            %   metricIdx - Index of the metric to visualize
            if nargin < 4, metricIdx = 1; end
            % 参数检查
            idx = find(params == -1);
            assert(numel(idx) == 2, 'The params needing to plot must be exactly 2 (-1).');
        
            % 提取固定参数和行索引
            fixid = (params ~= -1);
            fixparam = params(fixid);
            rowIndex = obj.findLike(obj.params(:,fixid), fixparam);
        
            % 提取需要绘制的数据
            data = [obj.params(1:obj.numGroups, ~fixid), obj.metrics(1:obj.numGroups, metricIdx)];
            data = data(rowIndex, :);
        
            % 拆分数据
            fp1 = data(:, 1); % 第一个变量
            fp2 = data(:, 2); % 第二个变量
            m = data(:, 3);   % 指标值

            % 创建网格
            [X, Y] = meshgrid(unique(fp1), unique(fp2));
            Z = zeros(size(X));
        
            % 填充指标值到网格中
            for i = 1:size(data, 1)
                xIdx = X(1, :) == fp1(i);
                yIdx = Y(:, 1) == fp2(i);
                Z(yIdx, xIdx) = m(i);
            end

             % 绘制三维柱状图
            figure;
            hBar  = bar3(Z);
            if nargin < 3
                pName = {['param',num2str(idx(1))],['param',num2str(idx(2))],'Metric'};
            end
            % 设置轴标签和标题
            xlabel(pName{1});
            ylabel(pName{2});
            zlabel(pName{3});

            % title(['3D Bar Plot of Metric ', num2str(metricIdx)]);
    
            % 设置网格和美化
            set(gca, 'XTickLabel', string(unique(fp1)), 'YTickLabel', string(unique(fp2)), ...
                'FontName', 'Times New Roman', 'FontSize', 12, 'FontWeight', 'normal');
            zlim([0, 1]);
            set(gca, 'ZTick', 0:0.2:1),
            grid on;
            colormap(jet); % 使用 jet 色图
            % setBarColor(hBar,1);
        end

        function [p,m] = plotEffect(obj, params, metricIdx)
            % Plots the effect of a single varying parameter on a selected metric
            % Input:
            %   params    - Vector with -1 for the parameter to vary, others fixed
            %   metricIdx - Metric index (1=acc, ..., 5=auc)
            % Output:
            %   p         - Sorted parameter values
            %   m         - Corresponding metric values
            if nargin < 3, metricIdx=1; end
            if metricIdx < 1 || metricIdx > 5
                error('Invalid metric index');
            end
            
            % 查找绘制参数的索引
            plotIdx = find(params == -1);
            if numel(plotIdx) ~= 1
                error('Exactly one parameter must be specified for plotting');
            end
            plotIdx = plotIdx(1);
            
        
            % 筛选满足固定条件的行
            data = obj.params(1:obj.numGroups,:);
            ipdata = data(:,params ~= -1);
            fixparam = params(:,params ~= -1);
            rowIndex = obj.findLike(ipdata,fixparam);

            plotparam = data(rowIndex,plotIdx);
            plotm = obj.metrics(rowIndex, metricIdx);
            
            % 如果没有符合条件的行，报错
            if isempty(plotparam)
                warning('No matching parameter sets found');
                cprintf('yellow','No matching parameter sets found\n');
                p = Inf;m=[];
                return;
            end
            
            % 绘制
            [p, sortOrder] = sort(plotparam);
            m = plotm(sortOrder);
            

            figure;
            plot(p, m, 'Marker', '.','MarkerSize',1 ,'LineWidth', 1.2);
            xlabel(['Param ' num2str(plotIdx)]);
            ylabel(['Metric ' num2str(metricIdx)]);
            xlim([min(p), max(p)]);  % 设置 x 轴的显示范围
            title(['Effect of Param ' num2str(plotIdx) ' on Metric ' num2str(metricIdx)]);
            % grid on;
        end
        
        function idx = findParam(obj, param)
            if obj.numGroups == 0
                idx = 0;
                return
            end
            % diff_matrix = abs(obj.params(1:obj.numGroups, :) - param);
            % % 查找完全匹配的参数组（所有列都相等）
            % matchIdx = all(diff_matrix < 1e-9, 2);  % 返回逻辑数组，表示每行是否完全匹配
            % % 如果找到了匹配的行，返回其索引，否则返回 -1
            % if ~(sum(matchIdx) ~= 1 || sum(matchIdx) ~= 0)
            %     error('错误，修改1e-9更小');
            % end
            % if any(matchIdx)
            %     idx = find(matchIdx);  % 返回匹配行的索引
            % else
            %     idx = 0;  % 没有找到匹配时返回 -1
            % end
            [~, idx] = ismember(param, obj.params, 'rows');
        end

        function exists = checkExist(obj, nparam)
            if obj.numGroups == 0
                exists = 0;
                return
            end
          
            [~, ia] = ismember(nparam, obj.params, 'rows');
            exists = all(ia > 0);  % 所有参数组都存在时返回 true
        end

        % 过滤功能：检查是否存在指定参数
        % function exists = checkExist(obj, nparam)
        %     exists = true;
        %     for pp=1:size(nparam,1)
        %         exists = exists & findParam(obj,nparam(pp,:));
        %         if ~exists 
        %             break
        %         end
        %     end
        % end
       
        function pToFile(obj,filename, metricIdx)
            % saveParamsToFile - 将参数和对应的指标写入文件，并按行排序
            % obj - ParameterTracker 对象
            % metricIdx - 指标索引: 1=acc, 2=sen, 3=spe, 4=f1, 5=auc
            % filename - 输出文件名
            if nargin < 3
                metricIdx = 1;
            end
            % 检查指标有效性
            if metricIdx < 1 || metricIdx > 5
                error('Invalid metric index');
            end
            
            % 获取所有参数和对应的指标
            param = obj.params(1:obj.numGroups,:);       % 所有参数
            metric = obj.metrics(1:obj.numGroups, metricIdx); % 选择指定指标
            
            % 将参数和指标合并为一个矩阵，方便排序
            data = [param, metric];
            
            % 排序
            [sortedData, ~] = sortrows(data);  % sortrows按所有列进行排序

            % 检查文件是否存在
            [parentDir, ~, ~] = fileparts(filename);
            if ~exist(parentDir, 'dir')
                % 如果父目录不存在，创建多级目录
                mkdir(parentDir);
            end

            % 打开文件进行写入
            fid = fopen(filename, 'w');
            if fid == -1
                error('无法打开文件');
            end
            
            % 打印标题
            fprintf(fid, 'Parameters and Metrics，Seeds: %s\n',num2str(obj.seeds));
            fprintf(fid, '-----------------------------------------------\n');
            
            % 写入每行数据，确保列宽一致，左对齐
            for i = 1:size(sortedData, 1)
                % 处理每一行的输出
                paramStr = sprintf([repmat('%-10g', 1, size(param, 2)), '%-10g'], sortedData(i, 1:end-1), sortedData(i, end));
                fprintf(fid, '%s\n', paramStr);
            end
            
            % 关闭文件
            fclose(fid);
            fprintf('数据已保存到文件: %s\n', filename);
        end

       % 将整个类实例保存为模型文件（.mat 文件）
        function save(obj, filename)
            % saveModel - 保存 ParameterTracker 类的实例为 .mat 文件
            % filename - 模型文件名（.mat 文件）
            [parentDir, ~, ~] = fileparts(filename);
            if ~exist(parentDir, 'dir')
                mkdir(parentDir);  % 创建父目录
            end

            save(filename, 'obj');
            fprintf('已将模型保存到文件: %s\n', filename);
        end
        
        % 从文件加载数据
        function obj = load(~,filename)
            % loadFromFile - 从文件加载 ParameterTracker 实例
            % filename - 文件名
            % 加载 .mat 文件
            data = load(filename);
            obj = data.obj;  % 恢复 ParameterTracker 实例
            fprintf('已从 MAT 文件加载数据: %s\n', filename);
        end

    end
end
