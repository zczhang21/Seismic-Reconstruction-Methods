%% =======================================================
% Spectrum comparison (Before vs After) - amplitude spectrum
% =======================================================

% 频率轴
nfft = 2^nextpow2(nt*2);           % 适当零填充让曲线更平滑
f = (0:nfft/2) / (nfft*dt);        % 单边频率 (Hz)

% 可选：只用"观测到的道"来算 observed 的频谱（避免缺失道的 0 拉低频谱）
mask_line = squeeze(T_spatial(:, idx2, idx3, idx4));   % n1 x 1 (对应 slice 的 mx 方向)
mask_line = mask_line(:) > 0;

% 加窗（减少谱泄漏）
w = hann(nt);

% --- True ---
X_true = fft( (slice_true .* w), nfft, 1 );            % nfft x n1
A_true = abs(X_true(1:nfft/2+1, :));
A_true_mean = mean(A_true, 2);

% --- Observed (建议只用观测到的道) ---
X_obs = fft( (slice_obs .* w), nfft, 1 );
A_obs = abs(X_obs(1:nfft/2+1, :));
if any(mask_line)
    A_obs_mean = mean(A_obs(:, mask_line), 2);
else
    A_obs_mean = mean(A_obs, 2);
end

% --- Reconstructed ---
X_rec = fft( (slice_rec .* w), nfft, 1 );
A_rec = abs(X_rec(1:nfft/2+1, :));
A_rec_mean = mean(A_rec, 2);

% 转 dB（相对最大值归一，更好比较形状）
eps0 = 1e-12;
A_true_db = 20*log10(A_true_mean / (max(A_true_mean)+eps0) + eps0);
A_obs_db  = 20*log10(A_obs_mean  / (max(A_obs_mean) +eps0) + eps0);
A_rec_db  = 20*log10(A_rec_mean  / (max(A_rec_mean) +eps0) + eps0);

% 修改为显示 0-100 Hz 的频谱范围
fmin_plot = 0; 
fmax_plot = 100;

figure('Name','Amplitude Spectrum Comparison','Color','w');
plot(f, A_true_db, 'LineWidth', 1.5); hold on;
plot(f, A_obs_db,  'LineWidth', 1.5);
plot(f, A_rec_db,  'LineWidth', 1.5);
grid on; xlim([fmin_plot fmax_plot]);
xlabel('Frequency (Hz)'); ylabel('Amplitude (dB, normalized)');
title(sprintf('Slice Spectrum (idx2,idx3,idx4) = (%d,%d,%d)', idx2, idx3, idx4));
legend('True','Observed (masked)','Reconstructed','Location','best');
