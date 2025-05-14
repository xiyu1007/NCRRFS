function colors = ColorMap(n,filename,Rec,Dot,SampleSV, min_sat, min_val, num_samples)
    if nargin < 1 n=10; end % 生成 10 个颜色 max =200
    if nargin < 2 filename=[]; end % 保存颜色
    if nargin < 3 Rec=0; end % 显示颜色
    if nargin < 4 Dot=0; end % 显示颜色
    if nargin < 5 SampleSV = 0; end
    if nargin < 6 min_sat = 1; end % 饱和度最小值
    if nargin < 7 min_val = 1; end % 亮度最小值
    if nargin < 8 num_samples = 5000; end % 采样点数量
    % 示例 color = ColorMap(20,'ColorMap.txt');

    colors = generate_distinguishable_colors(n, SampleSV, min_sat, min_val, num_samples);
    % 保存颜色到文件
    if ~isempty(filename)
        save_colors_to_file(colors, filename);
    end
    if Rec
        displayColorRec(colors);
    end
    if Dot
        displayColorBar(colors);
    end
end

function colors = generate_distinguishable_colors(n, SampleSV, min_sat, min_val, num_samples)
    % 默认值设置
    if nargin < 2 SampleSV = 0; end
    if nargin < 3 min_sat = 0.95; end
    if nargin < 4 min_val = 0.95; end
    if nargin < 5 num_samples = 5000; end
    if n > 200 n = 200; end
    % 生成 n 个易区分的颜色
    % 参数:
    %   n: 要生成的颜色数量
    %   min_sat: 饱和度的最小值
    %   min_val: 亮度的最小值
    %   num_samples: 采样点数量
    % 返回:
    %   colors: 生成的 n 个 RGB 颜色

    % 均匀分布的色调样本（0 到 1 之间）
    hue_samples = linspace(0, 1, num_samples);
    if SampleSV
        % 在 [min_sat, 1] 范围内随机生成饱和度样本
        sat_samples = min_sat + rand(1, num_samples) * (1 - min_sat);
        % 在 [min_val, 1] 范围内随机生成亮度样本
        val_samples = min_val + rand(1, num_samples) * (1 - min_val);
    else
        sat_samples = min_sat * ones(1, num_samples);
        val_samples = min_val * ones(1, num_samples);
    end
    % 初始化颜色矩阵
    rgb_colors = zeros(num_samples, 3);

    % 生成每个颜色的 RGB 值
    for i = 1:num_samples
        hsv_color = [hue_samples(i), sat_samples(i), val_samples(i)];
        rgb_colors(i, :) = hsv2rgb(hsv_color); % 转换为 RGB
    end

    % 贪心算法选择最易区分的 n 个颜色
    selected_colors = [];
    selected_indices = [];
    for i = 1:n
        if isempty(selected_colors)
            selected_indices(end+1) = 1; % 选择第一个颜色
        else
            % 计算与已选择颜色的最小距离
            min_distances = inf(num_samples, 1);
            for j = 1:size(selected_colors, 1)
                dists = sum((rgb_colors - selected_colors(j, :)).^2, 2);
                min_distances = min(min_distances, dists);
            end
            % 选择距离最大化的颜色
            [~, idx] = max(min_distances);
            selected_indices(end+1) = idx;
        end
        selected_colors = [selected_colors; rgb_colors(selected_indices(end), :)];
    end

    colors = rgb_colors(selected_indices, :); % 返回选择的 n 个颜色
end

function save_colors_to_file(colors, filename)
    % 将颜色保存到文本文件中
    % 参数:
    %   colors: 颜色数组，大小为 n x 3
    %   filename: 文件名（例如 'ColorMap.txt'）

    % 打开文件以写入
    fileID = fopen(filename, 'w');

    % 循环写入每个颜色的 RGB 值
    for i = 1:size(colors, 1)
        fprintf(fileID, '%.3f %.3f %.3f\n', colors(i, 1), colors(i, 2), colors(i, 3));
    end

    % 关闭文件
    fclose(fileID);
end

function displayColorRec(colors, num)
    % 此函数用于显示生成的颜色集合，每行显示不超过 num 个颜色。
    % 参数:
    %   colors: 颜色集合，大小为 n x 3，表示 n 个 RGB 颜色
    %   num: 每行最多显示的颜色数量
    if nargin <2 num=10; end
    n = size(colors, 1); % 获取颜色个数
    figure('Color', 'w'); % 创建新的图形窗口
    
    for i = 1:n
        row = floor((i-1) / num); % 当前颜色所在的行号
        col = mod(i-1, num); % 当前颜色在该行的位置
        % 绘制每个颜色的矩形条
        rectangle('Position', [col, -(row +1), 1, 1], 'FaceColor', colors(i, :), 'EdgeColor', 'none');
        hold on;
    end
    
    % 设置坐标轴比例
    axis equal;
    xlim([0, min(num, n)]); % x 轴范围，最多显示 num 个颜色宽度
    ylim([-ceil(n / num), 0]); % y 轴范围，根据行数调整
    set(gca, 'XTick', [], 'YTick', [], 'Box', 'on'); % 去掉坐标轴刻度和边框
    % 标题显示生成的颜色数量
    title(['Selected ', num2str(n), ' Distinguishable Colors']);
    axis off; % 隐藏轴线
    % 禁用默认交互功能
    disableDefaultInteractivity(gca);
    hold off;
end

function displayColorBar(colors)
    % 获取颜色数量
    numColors = size(colors, 1);
    % 绘制彩色条
    figure('Color', 'w')
    hold on;
    for i = 1:numColors
        rectangle('Position', [i-1, 0, 1, 1], 'FaceColor', colors(i, :), 'EdgeColor', 'none');
    end
    xlim([0, numColors]);
    ylim([0, 1]);
    axis off; % 隐藏轴线
    title('Color Visualization');
    disableDefaultInteractivity(gca);
    hold off;
end
