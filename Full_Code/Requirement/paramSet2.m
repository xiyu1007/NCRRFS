function combos = paramSet2(n,fix)
    if n <= 3 || (n - numel(fix) <=3)
        combos = Inf(1, n);
        return;
    end
    param = ones(1, n);
    fixed_num = n - 2;  % 每次固定2个变量
    indices = 1:n;

    fixed_combos = nchoosek(indices, fixed_num);
    num_combos = size(fixed_combos, 1);


    combos = Inf(num_combos, n);  % 初始化为double矩阵
    combos(:,fix) = 1;

    for i = 1:num_combos
        fixed_idx = fixed_combos(i, :);
        combos(i, fixed_idx) = param(fixed_idx);  % 将对应位置赋值为1
    end
end
