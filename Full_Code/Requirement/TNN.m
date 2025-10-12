function [G,f] = TNN(R,ref,dim,fhandle,p,tol,max_iter)
    % R : h * n * Views
    % dim in {0,1,2}
    if nargin < 6, tol = 1e-4; end
    if nargin < 7, max_iter = 100; end
    if nargin < 3 || isempty(dim), dim = 0; end
    % 调整维度
    R = shiftdim(R,dim);
    [~, ~, n3] = size(R);
    if isscalar(ref)
        ref = repmat(ref, 1, n3);
    end

    % FFT
    Rf = fft(R, [], 3); 
    Gf = zeros(size(Rf));

    f = 0;

    % 判断频率对称区间
    if mod(n3,2) == 0
        endSlice = n3/2 + 1;   % 偶数情况
    else
        endSlice = (n3+1)/2;   % 奇数情况
    end

    for i = 1:endSlice
        [U, S, V] = svd(Rf(:,:,i), 'econ');
        s = diag(S);

        if nargin >= 4
            for iter = 1:max_iter
                [~, dfs] = fhandle(s,p);
                s_new = max(s - ref(i)*dfs, 0);
                if norm(s_new - s, inf) < tol
                    break;
                end
                s = s_new;
            end
        else
            s_new = max(s - ref(i), 0);
        end

        % 更新目标函数值
        if nargout > 1
            if nargin >= 4
                f = f + sum(fhandle(s_new,p));
            else
                f = f + sum(s_new);
            end
        end

        % 更新当前频率块
        Gf(:,:,i) = U * diag(s_new) * V';

        % 共轭对称补全另一半频率块
        if i > 1
            if mod(n3,2) == 0 && i < n3/2 + 1
                % 偶数，补前半部分共轭
                Gf(:,:,n3-i+2) = conj(Gf(:,:,i));
            elseif mod(n3,2) == 1 && i <= (n3-1)/2
                % 奇数，补前半部分共轭
                Gf(:,:,n3-i+2) = conj(Gf(:,:,i));
            end
        end
    end

    % 偶数 Nyquist 频率必须实数
    if mod(n3,2) == 0
        Gf(:,:,n3/2+1) = real(Gf(:,:,n3/2+1));
    end

    % 逆 FFT 保证实数
    % G = ifft(Gf, [], 3);
    G = ifft(Gf, [], 3, 'symmetric');
    G = shiftdim(G,3-dim);
    G = squeeze(num2cell(G, [1 2]));
end
