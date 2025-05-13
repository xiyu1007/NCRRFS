function [X,Y] = getADData(DataPath,Group,num_from,filter,seed,trim)
    % X: n*d, Y: N*C
    if nargin < 2
        Group = {'AD','CN'};
    end
    % Group = upper(Group);
    if nargin < 3
        num_from = 14;
    end
    if nargin < 4
        filter = false;
    end
    

    for ii = 1:length(DataPath)
        data = readtable(DataPath{ii,:},'VariableNamingRule','preserve');
        % 将所有 NaN 值替换为 0
        % 遍历每一列
        for k = 1:width(data)
            if isnumeric(data{:,k})  % 如果是数值列
                data{:,k} = fillmissing(data{:,k}, 'constant', 0);  % 将 NaN 替换为 0
            elseif iscellstr(data{:,k}) || isstring(data{:,k})  % 如果是字符串列
                % 替换字符串列中的 NaN 为字符串 '0'
                data{:,k} = fillmissing(data{:,k}, 'constant', '0');
            end
        end

        if ii==1
            distribution(data);
        end
        data.Group(ismember(data.Group, 'SMC')) = {'MCI'};
        distribution(data);
        data = data(ismember(data.Group, Group), :); %()子表
        data = sortrows(data, 'Subject'); % 按 Subject 列排序

        if nargin >= 6 && trim
            if ii==1
                % 随机打乱行的顺序
                if nargin >=5
                    rng(seed);
                end
                randOrder = randperm(height(data));  % 获取随机的行索引
                rng('shuffle');
            end
            
            data = data(randOrder, :);  % 按随机顺序重新排列数据
            % data = data(ismember(data.Group, Group), :); %()子表
        
        
            g1 = data(ismember(data.Group, Group{1}), :);
            g2 = data(ismember(data.Group, Group{2}), :);
            n = min(height(g1),height(g2));
            g1 = g1(1:min(height(g1),n),:);
            g2 = g2(1:min(height(g2),n),:);
            data = [g1;g2];
        end

        if ii == 1
            n = size(data,1);
            Y = zeros(n,length(Group));
            for jj=1:n
                groupIndex  = strcmp(Group, data.Group{jj});
                Y(jj,:) = groupIndex;
            end  
            summaryTable = table();
            % 统计信息
            for j = 1:length(Group)
                groupData = data(strcmp(data.Group, Group{j}), :);
                groupCount = height(groupData);
                ageMean = mean(groupData.Age);
                ageStd = std(groupData.Age);
                femaleCount = sum(strcmp(groupData.Sex,'F'));
                maleCount = sum(strcmp(groupData.Sex,'M'));
                CDRMean = mean(groupData.CDR);
                CDRStd = std(groupData.CDR);
                mmseMean = mean(groupData.MMSE);
                mmseStd = std(groupData.MMSE);
                npiMean = mean(groupData.NPI_Q);
                npiStd = std(groupData.NPI_Q);
                gdsMean = mean(groupData.GDS);
                gdsStd = std(groupData.GDS);
                % 格式化 Age 为 "Mean ± Std"
                Gender = sprintf('%d/%d', femaleCount, maleCount);
                ageFormatted = sprintf('%.1f ± %.1f', ageMean, ageStd);
                mmseFormatted = sprintf('%.1f ± %.1f', mmseMean, mmseStd);
                cdrFormatted = sprintf('%.1f ± %.1f', CDRMean, CDRStd);
                npiFormatted = sprintf('%.1f ± %.1f', npiMean, npiStd);
                gdsFormatted = sprintf('%.1f ± %.1f', gdsMean, gdsStd);
                % 创建当前 Group 的统计表
                groupSummary = table({Group{j}}, {Gender}, {ageFormatted}, ...
                    {mmseFormatted},{cdrFormatted},{npiFormatted},{gdsFormatted}, ...
                     'VariableNames', {'Group', 'Gender', 'Age(Mean ± Std)','MMSE(Mean ± Std)','CDR(Mean ± Std)','NPI(Mean ± Std)','GDS(Mean ± Std)'});
                % 合并到总表
                summaryTable = [summaryTable; groupSummary];
            end
            % 将汇总结果写入 CSV 文件
            % writetable(summaryTable, 'GroupStats.xlsx');
            writetable(summaryTable, 'GroupStats.csv', 'Encoding', 'UTF-8');
            % writetable(summaryTable, 'GroupStats.xlsx','WriteMode','overwrite');
        end
        data = data{:,num_from:size(data,2)}; % {}数组
        % % 检查每一行是否全为 0
        % zeroID = all(data == 0, 1);
        % data(:,zeroID) = 1;
        if filter
            filterNum = 0.02;
            data = data(:,mean(data,1) > filterNum);
            fprintf("过滤后的数据维度：%d\n",size(data,2));
        end
        X{ii} = data;
    end

    % disp('-------------------------------------')
    % for i=1:numel(Group)
    %     fprintf('%-4s: %d\n',Group{i},sum(Y(:,i)));
    % end
    distribution(Y);
end

