
clear; clc; close all;

%% 1. 生成合成测试数据 (模拟 2D 交叉同相轴)
dt = 0.004;             % 时间采样率 (s)
nt = 256;               % 时间采样点数
t = (0:nt-1) * dt;      % 时间轴

dx = 25;                % 空间采样率 (m)
nx = 60;                % 总道数
x = (0:nx-1) * dx;      % 空间轴 (全集)

% 模拟两个倾角不同的同相轴
p_true1 = 0.0002;       % 射线参数 1 (s/m)
p_true2 = -0.00015;     % 射线参数 2 (s/m)
f_peak = 30;            % 雷克子波中心频率

% 生成全排列无缺失数据
d_full = zeros(nt, nx);
for ix = 1:nx
    tau1 = 0.3 + p_true1 * x(ix);
    tau2 = 0.5 + p_true2 * x(ix);
    
    % 雷克子波生成
    wavelet1 = (1 - 2 * (pi * f_peak * (t - tau1)).^2) .* exp(-(pi * f_peak * (t - tau1)).^2);
    wavelet2 = (1 - 2 * (pi * f_peak * (t - tau2)).^2) .* exp(-(pi * f_peak * (t - tau2)).^2);
    
    d_full(:, ix) = wavelet1 + wavelet2;
end

%% 2. 模拟空间规则缺失 (产生假频)
missing_ratio = 2; % 每隔2道保留1道 (50%缺失)
obs_idx = 1:missing_ratio:nx; 
x_obs = x(obs_idx);
nx_obs = length(x_obs);
d_obs = zeros(nt, nx);
d_obs(:, obs_idx) = d_full(:, obs_idx); % 缺失道补零用于显示
d_input = d_full(:, obs_idx);           % 实际输入算法的数据矩阵 (nt x nx_obs)

%% 3. 频率域抗混叠贪婪 Radon 变换 (AAGRT)
% 参数设置
p_max = 0.0004;           % 最大射线参数范围
dp = 0.000005;            % 射线参数扫描步长
p = -p_max:dp:p_max;      % Radon域 p 轴
np = length(p);

nitr = 10;                % 贪婪算法(匹配追踪)单频率迭代次数
N_keep = 1;               % 每次选取的最大能量个数 (按伪代码N, 这里设为1即标准MP)

% 步骤 1: 1D 傅里叶变换到 F-X 域
Nfft = nt;
D_FX = fft(d_input, Nfft, 1);
df = 1 / (Nfft * dt);
freqs = (0:Nfft/2) * df;  % 仅计算正频率

% 初始化权重 A(p) 和 模型
A_p = zeros(1, np);
M_FP = zeros(Nfft, np);          % 插值后: AAGRT Radon 模型
M_FP_Adjoint = zeros(Nfft, np);  % 插值前: 伴随 Radon 模型 (用于对比)
Weighted_ME_FP = zeros(Nfft, np); % 【新增】：用于保存在 F-P 域的加权伴随模型

% 步骤 2: 沿频率循环 (必须从低频到高频，以累积抗混叠权重)
% 避开直流分量(f=0)，从第二个频率开始
for idf = 2:(Nfft/2 + 1)
    w = 2 * pi * freqs(idf);
    
    % 当前频率的观测数据
    d_w = D_FX(idf, :).'; % 大小: nx_obs x 1
    
    % 正演与伴随算子矩阵 (局部)
    L = exp(1i * w * x_obs.' * p); 
    L_adj = L'; % 共轭转置
    
    d_resi = d_w;
    m_w = zeros(np, 1);
    
    % 初始伴随变换，用于更新抗混叠权重 A(p)
    m_e_init = L_adj * d_resi; 
    M_FP_Adjoint(idf, :) = m_e_init.'; % 【新增】保存插值前伴随模型
    A_p = A_p + abs(m_e_init).';       % 根据之前的低频累积能量更新权重
    % 【新增】：记录当前频率下，第一次进入贪婪迭代前的加权能量 (即 F-P 域的 weighted_me)
    Weighted_ME_FP(idf, :) = abs(m_e_init).' .* A_p;

    % 贪婪迭代 (Matching Pursuit)
    for iter = 1:nitr
        % 计算伴随模型残差
        m_e = L_adj * d_resi; 
        
        % 利用低频累积权重 A(p) 对当前伴随模型加权，压制假频
        weighted_me = abs(m_e).' .* A_p; 
        
        % 寻找加权后能量最强的 N_keep 个元素的索引
        [~, sort_idx] = sort(weighted_me, 'descend');
        best_idx = sort_idx(1:N_keep);
        
        for k = 1:N_keep
            idx = best_idx(k);
            alpha = (L(:, idx)' * d_resi) / nx_obs; 
            
            % 更新模型
            m_w(idx) = m_w(idx) + alpha;
            
            % 更新数据残差
            d_resi = d_resi - L(:, idx) * alpha;
        end
    end
    
    % 保存当前频率的优化模型
    M_FP(idf, :) = m_w.';
    
    % 由于是实数信号，负频率共轭对称
    if idf > 1 && idf < (Nfft/2 + 1)
        M_FP_Adjoint(Nfft - idf + 2, :) = conj(m_e_init.');
        M_FP(Nfft - idf + 2, :) = conj(m_w.');
    end
end

%% 4. 数据重构 (利用求解出的 Radon 模型向全空间网格插值)
D_FX_interp = zeros(Nfft, nx);

for idf = 2:(Nfft/2 + 1)
    w = 2 * pi * freqs(idf);
    % 全空间的傅里叶正演算子
    L_full = exp(1i * w * x.' * p);
    
    % 利用最优的 M_FP 预测全空间数据
    m_w = M_FP(idf, :).';
    D_FX_interp(idf, :) = (L_full * m_w).';
    
    if idf > 1 && idf < (Nfft/2 + 1)
        D_FX_interp(Nfft - idf + 2, :) = conj(D_FX_interp(idf, :));
    end
end

% IFFT 回到 T-X 域并取实部
d_interp = real(ifft(D_FX_interp, Nfft, 1));

%% 5. 绘图对比 (时间域)
figure('Position', [100, 100, 1000, 400], 'Name', 'Time Domain');
subplot(1, 4, 1);
imagesc(x, t, d_full); colormap(gray); caxis([-1 1]);
title('(a) Original Data'); xlabel('Offset (m)'); ylabel('Time (s)');
subplot(1, 4, 2);
imagesc(x, t, d_obs); colormap(gray); caxis([-1 1]);
title('(b) Input (50% missing)'); xlabel('Offset (m)');
subplot(1, 4, 3);
imagesc(x, t, d_interp); colormap(gray); caxis([-1 1]);
title('(c) Interpolated Data'); xlabel('Offset (m)');
subplot(1, 4, 4);
imagesc(x, t, d_full - d_interp); colormap(gray); caxis([-1 1]);
title('(d) Difference (a - c)'); xlabel('Offset (m)');

%% 6. 绘图对比 (Radon / F-P 域前后对比 - 包含 Weighted_ME)
% 截取绘图频带 0 - 80 Hz 以避免高频无用区域
f_max_plot = 80;
f_idx = freqs <= f_max_plot;
freqs_plot = freqs(f_idx);

abs_M_Adjoint = abs(M_FP_Adjoint(f_idx, :));
abs_Weighted_ME = abs(Weighted_ME_FP(f_idx, :)); % 【提取加权能量】
abs_M_AAGRT   = abs(M_FP(f_idx, :));

figure('Position', [100, 200, 1200, 450], 'Name', 'F-P Domain Progressive Comparison');

% 1. 原始伴随模型 (充满假频)
subplot(1, 3, 1);
vmax1 = max(abs_M_Adjoint(:)) * 0.8;
imagesc(p*1000, freqs_plot, abs_M_Adjoint); 
colormap(1-gray); caxis([0, vmax1]); set(gca, 'YDir', 'reverse');
title('(a) Adjoint Model (m_e)');
xlabel('Slowness p (ms/m)'); ylabel('Frequency (Hz)');

% 2. 加权后的伴随模型 (weighted_me) - 假频被压制
subplot(1, 3, 2);
% 加权后的数值量级变了，重新计算色标
vmax2 = max(abs_Weighted_ME(:)) * 0.8; 
imagesc(p*1000, freqs_plot, abs_Weighted_ME); 
colormap(1-gray); caxis([0, vmax2]); set(gca, 'YDir', 'reverse');
title('(b) Weighted Model (weighted\_me)');
xlabel('Slowness p (ms/m)');

% 3. AAGRT最终提取的稀疏模型 (干净的聚焦点)
subplot(1, 3, 3);
vmax3 = max(abs_M_AAGRT(:)) * 0.8;
imagesc(p*1000, freqs_plot, abs_M_AAGRT); 
colormap(1-gray); caxis([0, vmax3]); set(gca, 'YDir', 'reverse');
title('(c) Final Sparse Model (M_{FP})');
xlabel('Slowness p (ms/m)');

%% 7. 计算并标注信噪比 (SNR)

% 定义计算函数 (匿名函数)
calc_snr = @(orig, recon) 10 * log10(sum(orig(:).^2) / sum((orig(:) - recon(:)).^2));

% 1. 计算输入数据（含零值）的信噪比
% 注意：d_obs 中缺失道为0，这些0被视为误差的一部分
snr_input = calc_snr(d_full, d_obs);

% 2. 计算插值重构后的信噪比
snr_recon = calc_snr(d_full, d_interp);

% 打印结果到命令行
fprintf('\n=== 信噪比分析 ===\n');
fprintf('输入观测数据 (含缺失) SNR: %.2f dB\n', snr_input);
fprintf('AAGRT 插值重构后 SNR: %.2f dB\n', snr_recon);
fprintf('信噪比提升: %.2f dB\n', snr_recon - snr_input);

%% 8. 更新时间域绘图 (添加 SNR 标注)
figure('Position', [100, 100, 1200, 450], 'Name', 'Time Domain with SNR');

subplot(1, 4, 1);
imagesc(x, t, d_full); colormap(gray); caxis([-1 1]);
title('(a) Original Data'); xlabel('Offset (m)'); ylabel('Time (s)');

subplot(1, 4, 2);
imagesc(x, t, d_obs); colormap(gray); caxis([-1 1]);
title(sprintf('(b) Input\nSNR: %.2f dB', snr_input)); xlabel('Offset (m)');

subplot(1, 4, 3);
imagesc(x, t, d_interp); colormap(gray); caxis([-1 1]);
title(sprintf('(c) Interpolated\nSNR: %.2f dB', snr_recon)); xlabel('Offset (m)');

subplot(1, 4, 4);
imagesc(x, t, d_full - d_interp); colormap(gray); caxis([-1 1]);
title(sprintf('(d) Difference\n')); 
xlabel('Offset (m)');