classdef Recorder
    properties
        % 评估指标
        ACC        % 准确度
        SEN        % 灵敏度
        SPE        % 特异性
        AUC        % 曲线下面积
        F1         % F1分数
        
        % 训练数据
        Loss = {}      % 损失历史（如果有的话）
        FSID = {}      % 特征选择的ID
        nfs = -1;
        
        % 预测结果
        labs      % 模型预测
        decs       % 决策值
        
        % 实验参数
        params     % 固定的实验参数
        
        % 实验配置
        r          % 实验重复次数
        k          % k折交叉验证次数

        % seeds
        evaSeeds % 评估种子
        seeds % 实验种子

        runtime = 0 % 记录时间

        cr % 当前重复次数

        pt

        task = NaN;

        method = '';

        ins
    end
    
    methods
        % 构造函数，初始化所有属性
        function obj = Recorder(r, k, params)
            if nargin > 0
                obj.cr = 0;
                obj.r = r;
                obj.k = k;
                
                % 预分配内存，避免动态扩展
                obj.ACC = NaN(r, k);
                obj.SEN = NaN(r, k);
                obj.SPE = NaN(r, k);
                obj.AUC = NaN(r, k);
                obj.F1 = NaN(r, k);

                obj.evaSeeds = [];
                % obj.seeds = NaN(1,r);
                obj.seeds = [];

                % 预分配训练数据存储
                obj.Loss = cell(r,k);
                obj.FSID = cell(r,k);
                
                % 预分配预测结果存储
                obj.labs = cell(r,k);
                obj.decs = cell(r,k);
                
                % 初始化实验参数
                if nargin > 2
                    obj.params = params; % 设定固定的实验参数
                else
                    obj.params = []; % 默认无参数
                end
            end
        end
        
        function obj = log(obj, results)
            if iscell(results)
                for i=1:numel(results)
                    result = results{i};
                    obj.cr = obj.cr + 1;
                    cr_ = obj.cr;
                    obj.ACC(cr_, :) = result.acc;
                    obj.SEN(cr_, :) = result.sen;
                    obj.SPE(cr_, :) = result.spe;
                    obj.AUC(cr_, :) = result.auc;
                    obj.F1(cr_, :) = result.f1;
                    for ck = 1:obj.k
                        obj.labs{cr_,ck} = result.labs{ck};        
                        obj.decs{cr_,ck} = result.decs{ck}; 
                    end
                end
            else
                obj.cr = obj.cr + 1;
                cr_ = obj.cr;
                obj.ACC(cr_, :) = results.acc;
                obj.SEN(cr_, :) = results.sen;
                obj.SPE(cr_, :) = results.spe;
                obj.AUC(cr_, :) = results.auc;
                obj.F1(cr_, :) = results.f1;
                obj.evaSeeds = results.seeds; 
                for ck = 1:obj.k
                    obj.labs{cr_,ck} = results.labs{ck};        
                    obj.decs{cr_,ck} = results.decs{ck}; 
                end
            end
        end

        % 获取评估指标
        function [value, stdV] = getMetrics(obj,type)
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
            if any(isnan(data))
                waring('there is nan in data, use 0 to replace!\n')
            end
            data(isnan(data)) = 0;
            value = mean(data,2);
            stdV = std(data,[],2);
        end
        

        
        % 写入.mat文件
        function save(obj, filename,~)
            [parentDir, ~, ~] = fileparts(filename);
            if ~isfolder(parentDir) % ~exist(parentDir, 'dir')
                mkdir(fullfile(pwd,parentDir));  % 创建父目录
            end
            save(filename, 'obj');
            if nargin < 3
                fprintf('已将模型保存到文件: %s\n', filename);
            end
        end
        
        % 从.mat文件加载数据
        function obj = load(~,filename)
            data = load(filename);
            obj = data.obj;
            fprintf('已从 MAT 文件加载数据: %s\n', filename);
        end

        function [value, stdV] = getM(obj, type)
            % 根据 type 返回不同的性能指标的平均值和标准差
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
                    error('Invalid type. Valid types are: ACC, SEN, SPE, AUC, F1.');
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
            % 定义需要计算的指标类型
            metrics = {'ACC', 'SEN', 'SPE', 'F1', 'AUC'};
            
            % 打印表头
            fprintf('-----------------------------------------\n');
            fprintf('| Item \tMean\tStd \t|\n');
            fprintf('-----------------------------------------\n');
            
            % 遍历每种指标类型
            for i = 1:length(metrics)
                type = metrics{i};
                [value, stdV] = obj.getM(type); % 调用 getM 计算均值和标准差
                % 美观地打印结果
                fprintf('| %-4s \t%5.4f\t%5.4f\t|\n', type, value, stdV);
            end
            
            % 打印表尾
            fprintf('-----------------------------------------\n');
        end

        function [obj,flag] = update(obj,o1)
            [v, ~] = obj.getM('acc');
            [v1,~] = o1.getM('acc');
            flag = 0;
            if v < v1 
                obj = o1;
                flag = 1; 
            elseif abs(v - v1) < 1e-5
                [va, ~] = obj.getM('auc');
                [v1a,~] = o1.getM('auc');
                [vf, ~] = obj.getM('f1');
                [v1f,~] = o1.getM('f1');
                if va < v1a 
                    obj = o1;
                    flag = 1;
                elseif abs(va - v1a)< 1e-5 && vf < v1f
                    obj = o1;
                    flag = 1;
                elseif o1.nfs < obj.nfs && abs(va - v1a)< 1e-5 && abs(vf - v1f )< 1e-5
                    obj = o1;
                    flag = 1;
                end
            end
        end

        function [labslist,decslist,auc] = plotROC(obj, r, k)
            % 如果没有提供k，则合并r的所有k数据
            if nargin < 3, k = obj.k;end
            if nargin < 2, r = obj.r;end
            % 没有提供k，合并该r的所有k折数据
            labslist = [];
            decslist = [];
            for ridx  = 1:r
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
            % 使用合并的预测和决策值计算 ROC 曲线
            [fpr, tpr, ~, auc] = perfcurve(labslist(:), decslist(:), 1);
            
            % 绘制ROC曲线
            figure;
            % plot(fpr, tpr, 'o-', 'LineWidth', 1);
            plot(fpr, tpr, 'Color','r', 'LineWidth', 1);
            title(['ROC Curve for Experiment ' num2str(r)]);
            xlabel('False Positive Rate (FPR)');
            ylabel('True Positive Rate (TPR)');
            % grid off;
            % axis([0 1 0 1]);  % 设置坐标轴范围 [0, 1]
        end

        function loss = plotLoss(obj,be,r,k,color)
            if nargin < 2, be = 1; end
            if nargin < 3, r = 1; end
            if nargin < 4, k = 0; end
            if nargin < 5, color = ColorMap(10); end

            if k == 0
                loss = obj.Loss(r,:);
            else
               loss = obj.Loss(r,k);
            end

            figure;
            hold on;
            for i = 1:numel(loss)
                L = loss{i};      
                L = L(:,be:end);  
                len = size(L, 2);
                if iscell(color) 
                    c = color{i};
                elseif r ~= 0 && isscalar(color)
                    c = color;
                else
                    c = color(i,:);
                end
                plot(1:len, L, 'Color', c, ...
                    'Marker', 'o','MarkerSize',3 ,'LineWidth', 1, 'DisplayName', ['Loss ' num2str(i)]);
            end
            hold off; 
            legend show; 
            xlabel('Epoch'); 
            ylabel('Loss'); 
            title('Loss'); 
        end
    end
   
end
