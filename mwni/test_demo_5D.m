clear all;
close all;

% =======================================================
% Parameters to make synthetic example  
% =======================================================

 dt = 2/1000;                               % Sampling interval secs
 nt = 256;                                  % Number of samples in time
 Tmax = (nt-1)*dt;             
 f0 = 25.0;                                 % Central freq. of wavelet (hz)

% Parameters for superposition of 4 events 
% I=2 means parabolic 

 p1 =  [0.02, -0.1, 0.04,  0.09]*dt/300;            
 p2 =  [0.02,  0.1, 0.09, -0.04]*dt/300;
 p3 =  [0.01,  0.0, 0.02,  0.03]*dt/400;
 p4  = [0.04,  0.0, 0.01, -0.03]*dt/400;
 t0 = [Tmax/4., Tmax/3.,Tmax/2.,Tmax/1.4];
 d1 = 18; d2 = 18; d3 = 18; d4 = 18;  
 A = [-1, 1.2,-1.2, 1.2]; I = 2;
 
 n1 = 20; n2 =20; n3 =10; n4 =10;

% Make true tensor of data 

 [dr,t,mx,my,hx,hy] = linearevents5D(n1,n2,n3,n3,nt,d1,d2,d3,d4,dt,p1,p2,p3,p3,t0,A,f0,I);
 gam=1; % To control the jitter level of irregular data.
 [mx,my,hx,hy]=ndgrid(mx,my,hx,hy);
 [d0,t,mx_true,my_true,hx_true,hy_true]=linearevents5D_irregular(n1,n2,n3,n3,nt,d1,d2,d3,d4,dt,p1,p2,p3,p3,t0,A,f0,I,gam);
% add noise 

dobs = d0;     % dobs is 3D
                     % d1 is 3D
                     % d0 is 3D
disp(size(dobs));

SNR = 1; L = 5; 

dn = add_noise(dobs,SNR,L);
rand('seed',7)
erratic = randn(size(dn)); 
erratic = erratic.^5; 
erratic=erratic/max(erratic(:)); 
dn = dn + 3*erratic;

SNR_dB = snr(dobs, dn-dobs)

%是否包含噪声
dn = dobs;

% reshape the irregular data into a 2D matrix 
% in order to decimate the from the whole trace line (shot number).
% we don't need to decimate along two  X and Y directions
%% Decimation 
perc1  = 0.8;
ndec1 = floor(perc1*n1*n2*n3*n4);
rand('seed',7)
indx1 = sort(randperm(n1*n2*n3*n4, ndec1));


% dobs1 = reshape(dn,nt,n1*n2*n3*n4,1); % dobs1 is 2D
dobs1 = reshape(dn,nt,n1*n2*n3*n4,1); % dobs1 is 2D

dobs1(:,indx1) = zeros;                  % dobs1 is 2D

[T]  = GetSamplingOp(dobs1);

indxobs = setdiff([1:1:n1*n2*n3*n4]',indx1);   % define the index for binning
dobs111 = reshape(dobs1,nt,n1,n2,n3,n4);

%包含噪声的不规则
% filename = sprintf('data/5d_seg_irregular_%.1f_noise_%.2f.mat',perc1,SNR_dB);
% save(filename,'dr','dobs','dn','dobs111','mx_true','my_true','hx_true','hy_true')

%不包含噪声的不规则
filename = sprintf('data/5d_seg_irregular_%.1f_big.mat',perc1);
save(filename,'dr','dobs','dobs111','mx_true','my_true','hx_true','hy_true')

% %含噪声的规则
% filename = sprintf('data/5d_seg_regular_%.1f_noise_%.2f.mat',perc1,SNR_dB);
% save(filename,'dr','dn','dobs111','mx','my','hx','hy')

% %不含噪声的规则
% filename = sprintf('data/5d_seg_regular_%.1f_big.mat',perc1);
% save(filename,'dr','dobs111','mx','my','hx','hy')

kk = 5;
%figure;
%wigb([dr(:,:,kk,kk,kk),d0(:,:,kk,kk,kk),dn(:,:,kk,kk,kk),dobs111(:,:,kk,kk,kk)])
%wigb([dobs111(:,:,kk,kk,kk)])


%% =======================================================
% MWNI Reconstruction 
% =======================================================

fprintf('\n--------------------------------------------\n');
fprintf('Starting MWNI Reconstruction...\n');

% 1. 准备 MWNI 参数
% -------------------------------------------------------
% 频率范围：涵盖小波中心频率 f0=25Hz
fmin_mwni = 1.0; 
fmax_mwni = 60.0; 

% 维度信息
NDim = 5; 
% 空间网格大小 [n1, n2, n3, n4]
K_mwni = [n1, n2, n3, n4]; 

% 迭代次数
max_iter_int = 10 ; % 内部 CG 迭代次数
max_iter_ext = 5;  % 外部 IRLS 迭代次数

% 2. 构建空间采样算子 T (Spatial Mask)
% -------------------------------------------------------
% MWNI 在 5D 模式下，内部循环是对频率切片 (n1,n2,n3,n4) 进行处理。
% 因此我们需要传入一个 4D 的空间采样掩码，而不是 5D 的数据掩码。
% 由于数据是整道缺失（Whole trace missing），掩码随时间不变。

T_spatial = zeros(n1*n2*n3*n4, 1);
T_spatial(indxobs) = 1; % indxobs 来自 main 代码，标记了保留的道
T_spatial = reshape(T_spatial, n1, n2, n3, n4);

% 3. 执行 MWNI 重建
% -------------------------------------------------------
% 输入 dobs111 维度为 [nt, n1, n2, n3, n4]
tic;
[d_rec] = mwni(dobs111, T_spatial, dt, fmin_mwni, fmax_mwni, NDim, K_mwni, max_iter_int, max_iter_ext);
time_mwni = toc;
fprintf('MWNI reconstruction finished in %.2f seconds.\n', time_mwni);

% 4. 计算信噪比 (SNR)
% -------------------------------------------------------
% 计算整体数据的 SNR
diff = dr - d_rec; % dr 是无噪声的真实数据
snr_rec = 20 * log10(norm(dr(:)) / norm(diff(:)));
fprintf('Reconstruction SNR: %.2f dB\n', snr_rec);

%% =======================================================
% Visualization at Slice (:,:,5,5,5)
% =======================================================

% 定义切片位置
idx2 = 5; % m_y index
idx3 = 5; % h_x index
idx4 = 5; % h_y index

% 提取切片数据 (变为 nt x n1 的 2D 剖面)
slice_true = squeeze(dr(:, :, idx2, idx3, idx4));
slice_obs  = squeeze(dobs111(:, :, idx2, idx3, idx4));
slice_rec  = squeeze(d_rec(:, :, idx2, idx3, idx4));
slice_err  = slice_true - slice_rec;

figure('Name', 'MWNI Reconstruction Result (Slice 5,5,5)', 'Color', 'w', 'Position', [100, 100, 1200, 600]);


amx = max(abs(dobs1(:))) * 0.5;

% 1. 原始数据 (True)
subplot(1, 4, 1);
wigb(slice_true, amx);
title('True Data');
xlabel('Trace (mx)'); ylabel('Time (s)');



% 2. 观测数据 (Observed/Missing)
subplot(1, 4, 2);
wigb(slice_obs ,amx);
title(['Observed (Decimation: ' num2str(perc1*100) '%)']);
xlabel('Trace (mx)');
ylabel(''); % 节省空间

snr_slice = 20 * log10(norm(slice_true(:)) / norm(slice_err(:)));

% 3. 重建数据 (Reconstructed)
subplot(1, 4, 3);
wigb(slice_rec, amx);
title(['Reconstructed (SNR=' num2str(snr_slice, '%.1f') 'dB)']);
xlabel('Trace (mx)');
ylabel('');

% 4. 误差 (Difference)
subplot(1, 4, 4);
wigb(slice_err, amx);
title('Reconstruction Error');
xlabel('Trace (mx)');
ylabel('');

% 可选：并在命令窗口输出特定切片的 SNR
% snr_slice = 20 * log10(norm(slice_true(:)) / norm(slice_err(:)));
% fprintf('Slice (:,:,%d,%d,%d) SNR: %.2f dB\n', idx2, idx3, idx4, snr_slice);
