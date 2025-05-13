function combos = paramSet2(n,fix)
    % Check if the number of parameters is too small or the number of non-fixed parameters is too small
    if n <= 3 || (n - numel(fix) <= 3)
        % Return an infinite matrix to indicate an invalid or trivial case
        combos = Inf(1, n);
        return;
    end
    % Initialize a parameter vector with all elements set to 1
    param = ones(1, n);
    % Calculate the number of parameters to be fixed in each combination
    fixed_num = n - 2;  % Fix all but 2 parameters
    % Generate indices for all parameters
    indices = 1:n;

    % Generate all combinations of fixed parameters
    fixed_combos = nchoosek(indices, fixed_num);
    % Get the number of combinations
    num_combos = size(fixed_combos, 1);

    % Initialize the combos matrix with infinite values
    combos = Inf(num_combos, n);  % Initialize as a double matrix
    % Set the fixed parameters to 1 in all combinations
    combos(:,fix) = 1;

    % Iterate over each combination
    for i = 1:num_combos
        % Get the indices of the fixed parameters for the current combination
        fixed_idx = fixed_combos(i, :);
        % Set the corresponding positions in the combos matrix to 1
        combos(i, fixed_idx) = param(fixed_idx);  % Assign 1 to the fixed parameters
    end
end