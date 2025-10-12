function colors = checkColor(colors,n,color_path)
    if nargin >= 3 && exist(color_path,'file')
        % 使用 readmatrix 读取数据
        colors = readmatrix(color_path);
        if any(colors > 1)
            colors = colors ./ 255;
        end
        return
    end
    if isempty(colors) || ...
       (iscell(colors) && numel(colors) <= n) || ...
       (isnumeric(colors) && size(colors, 1) <= n)
        
        colors = ColorMap(n + 1);
    end
end

