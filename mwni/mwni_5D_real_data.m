% Script for 5D Seismic Reconstruction using Patching + MWNI
%
%% 0.0) Preliminaries
clear; clc; close all;

% 1. Load 5D data and Parameters
disp('Loading incomplete 5D data from .mat file...');
load('..\CodeForBinning\data_binned.mat', 'data_binned'); 
% 假设 data_binned 是含有缺失道的观测数据 (dobs111)
raw_data = data_binned; 

[nt, n1, n2, n3, n4] = size(raw_data);
disp(['Data dimensions (nt, n1, n2, n3, n4): ', num2str([nt, n1, n2, n3, n4])]);

% =======================================================
% 2. 准备 MWNI 参数与全局掩码 (Mask)
% =======================================================
fmin_mwni = 1.0; 
fmax_mwni = 60.0; 
NDim = 5; 
max_iter_int = 10; 
max_iter_ext = 5;  
dt = 0.002;

% 构建全局的 4D 空间采样算子 T_spatial
% (假设 indxobs 已经存在工作区中，或者从 data_binned 中推导：非零道即为观测道)
% 这里提供一个稳健的推导方式：如果整道能量为0，则认为是缺失道
trace_energy = squeeze(sum(abs(raw_data), 1)); 
T_spatial_global = zeros(n1, n2, n3, n4);
T_spatial_global(trace_energy > 0) = 1; 

% ⚠️ 核心技巧：将 4D 掩码复制扩展为 5D，以便和数据使用同一套参数切块
mask_5D = repmat(reshape(T_spatial_global, [1, n1, n2, n3, n4]), [nt, 1, 1, 1, 1]);

% =======================================================
% 3. 设置分块参数 (Patching Parameters)
% =======================================================
PATCH.dt = dt; 
PATCH.it_WL  = 200;  PATCH.it_WO  = 40;
PATCH.ix1_WL = 20;   PATCH.ix1_WO = 10;
PATCH.ix2_WL = 24;   PATCH.ix2_WO = 8;
PATCH.ix3_WL = 12;   PATCH.ix3_WO = 0;
PATCH.ix4_WL = 8;    PATCH.ix4_WO = 0;

PATCH.x1min = 1; PATCH.x1max = n1;
PATCH.x2min = 1; PATCH.x2max = n2;
PATCH.x3min = 1; PATCH.x3max = n3;
PATCH.x4min = 1; PATCH.x4max = n4;

% =======================================================
% 4. 执行分块 (Patching Data & Mask)
% =======================================================
disp('Patching incomplete data...');
[PPatches, Minval, Maxval] = SeisPatch(raw_data, PATCH);

disp('Patching spatial mask...');
[MaskPatches, ~, ~] = SeisPatch(mask_5D, PATCH);

% 初始化一个用于存放重建结果的矩阵 (与 PPatches 维度一致)
PPatches_rec = zeros(size(PPatches));

% 获取切块后的总块数
% 假设 PPatches 的维度是 [it_WL, ix1_WL, ix2_WL, ix3_WL, ix4_WL, num_patches]
sz_p = size(PPatches);
num_patches = prod(sz_p(6:end)); % 计算第6维及以后的总元素个数（即总Patch数）

% 获取局部的空间网格大小 K_mwni_local
K_mwni_local = [sz_p(2), sz_p(3), sz_p(4), sz_p(5)];

% =======================================================
% 5. 在局部块中循环执行 MWNI
% =======================================================
disp(['Starting MWNI reconstruction on ', num2str(num_patches), ' patches...']);
tic;

% 如果你有 Parallel Computing Toolbox，可以将 for 换成 parfor 加速
for i = 1:num_patches
    % 提取第 i 个数据块 (5D: t, x1, x2, x3, x4)
    local_dobs = PPatches(:,:,:,:,:,i);
    
    % 提取第 i 个掩码块，并降维回 4D
    % (因为同一块内所有时间采样的空间掩码是一样的，取第1个时间切片即可)
    local_mask_5d = MaskPatches(:,:,:,:,:,i);
    local_T_spatial = squeeze(local_mask_5d(1, :, :, :, :)); 
    
    % 如果这个块是全空的（完全没有观测数据），则跳过重建，避免报错或除零
    if sum(local_T_spatial(:)) == 0
        PPatches_rec(:,:,:,:,:,i) = local_dobs;
        continue;
    end
    
    % 执行局部 MWNI 重建
    [local_drec] = mwni(local_dobs, local_T_spatial, dt, fmin_mwni, fmax_mwni, ...
                        NDim, K_mwni_local, max_iter_int, max_iter_ext);
                    
    % 保存重建后的块
    PPatches_rec(:,:,:,:,:,i) = local_drec;
    
    % 打印进度
    if mod(i, 10) == 0
        fprintf('Processed patch %d / %d\n', i, num_patches);
    end
end
time_mwni = toc;
fprintf('All patches MWNI reconstruction finished in %.2f seconds.\n', time_mwni);

% =======================================================
% 6. 重组数据 (Unpatching)
% =======================================================
disp('Unpatching reconstructed data...');
reconstructed_data = SeisUnPatch(PPatches_rec, nt, Minval, Maxval, PATCH);

% =======================================================
% 7. 结果质控 (Visualization & QC)
% =======================================================
disp('Generating QC plots...');
mid2 = round(n2/2); mid3 = round(n3/2); mid4 = round(n4/2);

figure('Name', '5D MWNI Reconstruction QC', 'Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
imagesc(raw_data(:, :, mid2, mid3, mid4));
title(sprintf('Incomplete Data (:,:,%d,%d,%d)', mid2, mid3, mid4));
colormap(gray); xlabel('n1'); ylabel('Time');

subplot(1, 3, 2);
imagesc(reconstructed_data(:, :, mid2, mid3, mid4));
title('Reconstructed Data (MWNI)');
colormap(gray); xlabel('n1'); ylabel('Time');

subplot(1, 3, 3);
residual = reconstructed_data(:, :, mid2, mid3, mid4) - raw_data(:, :, mid2, mid3, mid4);
% 理论上，在观测到的数据点上，残差应该接近于 0 (Data Fidelity)
imagesc(residual);
title('Difference (Rec - Obs)');
colormap(gray); colorbar;
xlabel('n1'); ylabel('Time');
% =======================================================
% 8. 保存重建结果到 .mat 文件
% =======================================================
disp('Saving reconstructed data to .mat file...');

% 定义输出文件名
output_filename = 'rec_5D_mwni.mat';

% 使用 -v7.3 格式保存重构后的数据变量 (reconstructed_data)
% 建议同时将维度参数也保存下来，方便日后读取时核对
save(output_filename, 'reconstructed_data', 'nt', 'n1', 'n2', 'n3', 'n4', 'dt', '-v7.3');

disp(['Data successfully saved as: ', output_filename]);