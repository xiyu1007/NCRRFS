classdef ParameterTracker
    properties
        params % 用于存储参数，每一行为一组参数
        metrics    % 用于存储每组参数对应的指标，行数与Parameters相同
        seeds      % 一个全局种子，所有参数组共享
        numGroups  % 当前参数组的数量
        maxGroups  % 最大支持的参数组数量
        numP
        method
    end
    
    methods
        % 构造函数：初始化
        function obj = ParameterTracker(NumP,MaxGroups)
            if nargin < 2
                MaxGroups = 1000; % 默认最大支持1000组参数
            end
            obj.numP = NumP;
            obj.maxGroups = MaxGroups;
            obj.numGroups = 0;
            obj.params = zeros(MaxGroups, NumP); % 每组参数
            obj.metrics = zeros(MaxGroups, 5);    % 5个指标：acc, sen, spe, f1, auc
            obj.seeds = NaN;                      % 固定的种子值
        end

        % 动态增加一组参数
        function [obj,acc] = addParameters(obj, param_values,result)
            % cell 或者 数组
            if iscell(result)
                mes = zeros(numel(result),5);
                for r = 1:numel(result)
                    re = result{r};
                    tm = [
                        mean(re.acc,'omitnan'), ... 
                        mean(re.sen,'omitnan'), ...
                        mean(re.spe,'omitnan'), ...
                        mean(re.f1,'omitnan'),  ...
                        mean(re.auc,'omitnan')  ...
                    ];
                    mes(r,:) = tm;
                end
                m = mean(mes, 1, 'omitnan');
            else
                m = result;
            end
            acc = m(1);
            idx = findParam(obj,param_values);
            if idx ~= 0
                obj.params(idx, :) = param_values; % 直接通过索引插入数据
                obj.metrics(idx, :) = m;
            else
                if obj.numGroups >= obj.maxGroups
                    % 自动扩容
                    obj.maxGroups = obj.maxGroups * 2; % 每次扩容时将容量翻倍
                    obj.params(obj.numGroups + 1:obj.maxGroups, :) = 0; % 扩展内存
                    obj.metrics(obj.numGroups + 1:obj.maxGroups, :) = 0; 
                end
                obj.numGroups = obj.numGroups + 1;
                obj.params(obj.numGroups, :) = param_values; % 直接通过索引插入数据
                obj.metrics(obj.numGroups, :) = m; % 直接通过索引插入数据
            end
        end
        
        % 为新增的参数组记录指标值
        function obj = addMetrics(obj, metrics)
            if obj.numGroups == 0
                error('没有参数组，请先添加参数组');
            end
            obj.metrics(obj.numGroups, :) = metrics; % 直接通过索引插入数据
        end

        function rowIndex = findLike(~,ipdata,param)
            rowIndex = ismember(ipdata, param, 'rows'); 
        end

        function [maxlist,maxparam] = plotAblation(obj,colors,leng,metricIdx)
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
                % 找到最后一列的最大值及其索引
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
            % PLOTPARAMBAR 绘制柱状图，展示两个参数组合下的指定指标的分布
            % params 需要包含 nfs !!!
            % 输入参数：
            %   obj        - 包含参数和指标的对象，具有属性 params 和 metrics
            %   params     - 参数数组，其中 -1 表示需要绘制的两个参数
            %   metricIdx  - 指定的指标列索引，用于绘图
            %
            % 示例：
            %   params = [-1, -1, 0.5, 0.6, 0.7, 0.8];
            %   metricIdx = 1;
            %   pName = {'𝛼', '𝛽', 'acc'};
            %   obj.plotParamBar(params,pName,metricIdx);
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

        function [p,m] = plotEffect(obj, params, metricIdx,draw)
            % plotEffect - 绘制指定参数(需要传入nfs)对某个指标的影响
            % params 需要包含 nfs !!!
            % params - 包含 -1 (绘制) 和固定值 (其他参数) 的数组
            % metricIdx - 指标索引: 1=acc, 2=sen, 3=spe, 4=f1, 5=auc
            % 验证输入
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
                p = Inf;m=[];
                return;
            end
            
            % 绘制
            [p, sortOrder] = sort(plotparam);
            m = plotm(sortOrder);
            
            if ~draw
                return
            end
            % figure;
            x = 1:1:numel(p);
            plot(x, m, 'Marker', '.','MarkerSize',2 ,'LineWidth', 1.2);

            set(gca, 'XTick', x) % 刻度位置
            set(gca, 'XTickLabel', p) % 刻度标签
            % xlabel(['Param ' num2str(plotIdx)]);
            % ylabel(['Metric ' num2str(metricIdx)]);
            xlabel(['Param ' num2str(plotIdx)]);
            if nargin >=4
                xlabel(label{1});
                ylabel(label{2});
            end
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
