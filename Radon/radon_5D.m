clear; clc; close all;
%% 1. 生成合成测试数据 (模拟 3D 交叉同相轴)
dt = 0.004;             % 时间采样率 (s)
nt = 200;               % 时间采样点数
t = (0:nt-1) * dt;      % 时间轴
dx = 25; dy = 25;       % 空间采样率 (m)
nx = 30; ny = 30;       % X和Y方向总道数 (30x30 = 900道)
x = (0:nx-1) * dx;      
y = (0:ny-1) * dy;      

% 生成 3D 空间网格并展平为 1D 向量
[X_grid, Y_grid] = ndgrid(x, y);
X_full = X_grid(:);     % 大小: (nx*ny) x 1
Y_full = Y_grid(:);     % 大小: (nx*ny) x 1
nx_full = length(X_full);

% 模拟两个具有 3D 倾角的同相轴
px_true1 = 0.0002;  py_true1 = 0.0001;   % 同相轴1的视慢度
px_true2 = -0.00015; py_true2 = 0.00015; % 同相轴2的视慢度
f_peak = 30;            % 雷克子波中心频率

% 生成全排列无缺失数据 (二维矩阵: nt x 全空间道数)
d_full = zeros(nt, nx_full);
for ixy = 1:nx_full
    tau1 = 0.3 + px_true1 * X_full(ixy) + py_true1 * Y_full(ixy);
    tau2 = 0.5 + px_true2 * X_full(ixy) + py_true2 * Y_full(ixy);
    
    wavelet1 = (1 - 2 * (pi * f_peak * (t - tau1)).^2) .* exp(-(pi * f_peak * (t - tau1)).^2);
    wavelet2 = (1 - 2 * (pi * f_peak * (t - tau2)).^2) .* exp(-(pi * f_peak * (t - tau2)).^2);
    
    d_full(:, ixy) = wavelet1 + wavelet2;
end

%% 2. 模拟空间随机缺失 (产生3D假频)
missing_ratio = 0.5; % 随机缺失 50% 的地震道
rng(42); % 固定随机种子以保证结果可复现
obs_idx = randperm(nx_full, round(nx_full * (1 - missing_ratio)));
X_obs = X_full(obs_idx);
Y_obs = Y_full(obs_idx);
nx_obs = length(obs_idx);

d_obs_flat = zeros(nt, nx_full);
d_obs_flat(:, obs_idx) = d_full(:, obs_idx); % 缺失道补零用于显示
d_input = d_full(:, obs_idx);                % 实际输入算法的数据矩阵 (nt x nx_obs)

%% 3. 频率域抗混叠贪婪 Radon 变换 (AAGRT - 3D 展平版)
% 参数设置
p_max = 0.0003;           
dp = 0.00002;             % 为了控制矩阵大小，步长略调大
p_scan = -p_max:dp:p_max; % 单维度 p 轴
Np_1d = length(p_scan);   % 单维度的射线参数个数
[Px_grid, Py_grid] = ndgrid(p_scan, p_scan);
Px = Px_grid(:).';        % 展平为行向量: 1 x np
Py = Py_grid(:).';        % 展平为行向量: 1 x np
np = length(Px);          % 总的 3D 慢度组合数
nitr = 15;                % 贪婪算法单频率迭代次数
N_keep = 1;               % 每次选取的最大能量个数

% 步骤 1: 1D 傅里叶变换到 F-X 域
Nfft = nt;
D_FX = fft(d_input, Nfft, 1);
df = 1 / (Nfft * dt);
freqs = (0:Nfft/2) * df;  

% 初始化权重 A(p) 和 模型
A_p = zeros(1, np);
M_FP = zeros(Nfft, np);          
M_FP_Adjoint = zeros(Nfft, np);  

% 步骤 2: 沿频率循环 (从低频到高频)
for idf = 2:(Nfft/2 + 1)
    w = 2 * pi * freqs(idf);
    d_w = D_FX(idf, :).'; % nx_obs x 1
    
    % 【核心适配】：用 3D 相位差公式，配合展平向量构建 2D 算子矩阵
    L = exp(1i * w * (X_obs * Px + Y_obs * Py)); 
    L_adj = L'; 
    
    d_resi = d_w;
    m_w = zeros(np, 1);
    
    % 初始伴随变换
    m_e_init = L_adj * d_resi; 
    M_FP_Adjoint(idf, :) = m_e_init.'; 
    A_p = A_p + abs(m_e_init).';       
    
    % 贪婪迭代 (OMP / MP)
    for iter = 1:nitr
        m_e = L_adj * d_resi; 
        weighted_me = abs(m_e).' .* A_p; 
        
        [~, sort_idx] = sort(weighted_me, 'descend');
        best_idx = sort_idx(1:N_keep);
        
        for k = 1:N_keep
            idx = best_idx(k);
            alpha = (L(:, idx)' * d_resi) / nx_obs; 
            m_w(idx) = m_w(idx) + alpha;
            d_resi = d_resi - L(:, idx) * alpha;
        end
    end
    
    M_FP(idf, :) = m_w.';
    
    if idf > 1 && idf < (Nfft/2 + 1)
        M_FP_Adjoint(Nfft - idf + 2, :) = conj(m_e_init.');
        M_FP(Nfft - idf + 2, :) = conj(m_w.');
    end
end

%% 4. 数据重构 (向 3D 全空间网格插值)
D_FX_interp = zeros(Nfft, nx_full);
for idf = 2:(Nfft/2 + 1)
    w = 2 * pi * freqs(idf);
    L_full = exp(1i * w * (X_full * Px + Y_full * Py));
    
    m_w = M_FP(idf, :).';
    D_FX_interp(idf, :) = (L_full * m_w).';
    
    if idf > 1 && idf < (Nfft/2 + 1)
        D_FX_interp(Nfft - idf + 2, :) = conj(D_FX_interp(idf, :));
    end
end
d_interp_flat = real(ifft(D_FX_interp, Nfft, 1));

%% 5. 绘图对比 (抽取 3D 数据体的一个 Inline 切片显示)
D_FULL_3D = reshape(d_full, [nt, nx, ny]);
D_OBS_3D = reshape(d_obs_flat, [nt, nx, ny]);
D_INTERP_3D = reshape(d_interp_flat, [nt, nx, ny]);
slice_idx = round(ny / 2);
slice_full = squeeze(D_FULL_3D(:, :, slice_idx));
slice_obs = squeeze(D_OBS_3D(:, :, slice_idx));
slice_interp = squeeze(D_INTERP_3D(:, :, slice_idx));

figure('Position', [100, 100, 1000, 400], 'Name', '3D Data - Inline Slice');
subplot(1, 4, 1); imagesc(x, t, slice_full); colormap(gray); caxis([-1 1]);
title(sprintf('(a) Original (Y=%.0fm)', y(slice_idx))); xlabel('X Offset (m)'); ylabel('Time (s)');
subplot(1, 4, 2); imagesc(x, t, slice_obs); colormap(gray); caxis([-1 1]);
title('(b) Input (50% missing)'); xlabel('X Offset (m)');
subplot(1, 4, 3); imagesc(x, t, slice_interp); colormap(gray); caxis([-1 1]);
title('(c) Interpolated'); xlabel('X Offset (m)');
subplot(1, 4, 4); imagesc(x, t, slice_full - slice_interp); colormap(gray); caxis([-1 1]);
title('(d) Difference (a - c)'); xlabel('X Offset (m)');

%% 6. 绘制 M_FP 在 f-px 域的结果图 (匹配参考图极简白底黑线风格)
% 截取绘图频带 0 - 80 Hz
f_max_plot = 80;
f_idx = freqs <= f_max_plot;
freqs_plot = freqs(f_idx);
num_f = sum(f_idx);

% 将一维平展的 P 轴重塑回 (Px, Py) 网格，并提取 f-px 投影
M_AAGRT_3D = reshape(abs(M_FP(f_idx, :)), [num_f, Np_1d, Np_1d]);
M_AAGRT_Px = max(M_AAGRT_3D, [], 3); % 沿 Py 维度取最大值投影到 Px

figure('Position', [300, 150, 400, 500], 'Color', 'w');
% 绘制图像，X轴单位转换为 ms/m
imagesc(p_scan * 1000, freqs_plot, M_AAGRT_Px);

% 设置纯黑白配色 (白底黑字，能量越强颜色越深)
colormap(1 - gray); 
% 或者使用 colormap(flipud(gray)); 效果相同

% 坐标轴设置：Y轴翻转使0在最上方，匹配参考图习惯
set(gca, 'YDir', 'reverse', 'TickDir', 'in');

% 限制 X 轴显示范围 (匹配你截图中的 -0.4 到 0.4)
xlim([-0.4, 0.4]);

title('Final Sparse M_{FP} (f-p_x)');
xlabel('Slowness p (ms/m)');
% 参考图中没有 Y 轴 Label，这里为了美观可以选择不加 ylabel('Frequency (Hz)')
%% 7. 3D 全数据体信噪比 (SNR) 计算

% 定义 3D 信噪比计算闭包
% SNR = 10 * log10( ||Signal||^2 / ||Noise||^2 )
calc_snr_3d = @(orig, recon) 10 * log10(sum(orig(:).^2) / sum((orig(:) - recon(:)).^2));

% 计算输入观测数据（含 50% 置零空洞）的 SNR
snr_input_3d = calc_snr_3d(d_full, d_obs_flat);

% 计算 AAGRT 插值重构后的 SNR
snr_recon_3d = calc_snr_3d(d_full, d_interp_flat);

% 命令行打印详细指标
%fprintf('\n' + repmat('=',1,30) + '\n');
fprintf('3D AAGRT 插值性能分析：\n');
fprintf('输入观测数据 SNR: %.2f dB\n', snr_input_3d);
fprintf('重构后全数据体 SNR: %.2f dB\n', snr_recon_3d);
fprintf('信噪比纯增益: %.2f dB\n', snr_recon_3d - snr_input_3d);
fprintf(repmat('=',1,30) + '\n');


