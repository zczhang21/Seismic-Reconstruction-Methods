function [dout] = mwni(data_binned, T, dt, fmin, fmax, NDim, K, max_iter_int, max_iter_ext)
% MWNI: reconstruction of binned data using Minimum Weighted Norm Interpolation

  if NDim == 1; error('Your data needs to have at least 2 dimensions, time and one spatial coordinate');end
  if NDim == 2;  [nt,n1]              = size(data_binned); N = n1; end
  if NDim == 3;  [nt,n1,n2]           = size(data_binned); N = [n1,n2]; end
  if NDim == 4;  [nt,n1,n2,n3]        = size(data_binned); N = [n1,n2,n3]; end
  if NDim == 5;  [nt,n1,n2,n3,n4]     = size(data_binned); N = [n1,n2,n3,n4];end

% Number of temporal frequencies of the DFT (20% padding)
  nfft = 2*floor(0.5*nt*1.2);

% Evaluate 1D DFT along time for all traces in the volume data_binned
  D = fft(data_binned,nfft,1); 

% Indeces of max and min frequencies for the DFT
  ifreq_min = floor(fmin*dt*nfft) + 1;
  ifreq_max = floor(fmax*dt*nfft) + 1;
  
  fprintf('Processing %dD data: %d frequencies from %.0f to %.0f Hz\n', ...
          NDim, ifreq_max-ifreq_min+1, fmin, fmax);

% Set initial volumes according to NDim
  if NDim == 2;  k1 = K;                                    x0 = zeros(k1,1);           Dout = zeros(nfft,n1); end;
  if NDim == 3;  k1 = K(1); k2 = K(2);                      x0 = zeros(k1,k2);          Dout = zeros(nfft,n1,n2); end;
  if NDim == 4;  k1 = K(1); k2 = K(2); k3 = K(3);           x0 = zeros(k1,k2,k3);       Dout = zeros(nfft,n1,n2,n3); end;
  if NDim == 5;  k1 = K(1); k2 = K(2); k3 = K(3); k4=K(4);  x0 = zeros(k1,k2,k3,k4);    Dout = zeros(nfft,n1,n2,n3,n4); end;

% ------------------------------------------
% ------------    2D case  -----------------
% ------------------------------------------

 if NDim == 2; 
  for ifreq = ifreq_min:ifreq_max
    % 修正1: 使用正确的频率索引
    y = squeeze(D(ifreq,:)).';  % 转置为列向量
    
    % 修正2: 传入正确的参数（N而不是M）
    [x,misfit] = mwni_irls(y, x0, T.', NDim, N, K, max_iter_int, max_iter_ext, 1);
    % 用上一时刻的值
    x0 = x;
    % 修正3: 重建时使用ones采样算子，并只取前N个点
    y_recon = operator_nfft(x, ones(n1,1), NDim, N, K, 1);
    
    % 修正4: 正确赋值维度
    Dout(ifreq,:) = y_recon.';
  end
 end;

% ------------------------------------------
% ------------    3D case  -----------------
% ------------------------------------------

 if NDim == 3; 
  for ifreq = ifreq_min:ifreq_max
    y = squeeze(D(ifreq,:,:));
    [x,misfit] = mwni_irls(y, x0, T, NDim, N, K, max_iter_int, max_iter_ext, 1);
    y_recon = operator_nfft(x, ones(n1,n2), NDim, N, K, 1);
    Dout(ifreq,:,:) = y_recon;
  end
 end;

% ------------------------------------------
% ------------    4D case  -----------------
% ------------------------------------------

 if NDim == 4; 
  for ifreq = ifreq_min:ifreq_max
    y = squeeze(D(ifreq,:,:,:));
    [x,misfit] = mwni_irls(y, x0, T, NDim, N, K, max_iter_int, max_iter_ext, 1);
    y_recon = operator_nfft(x, ones(n1,n2,n3), NDim, N, K, 1);
    Dout(ifreq,:,:,:) = y_recon;
  end
 end;

% ------------------------------------------
% ------------    5D case  -----------------
% ------------------------------------------

 if NDim == 5; 
  for ifreq = ifreq_min:ifreq_max
    y = squeeze(D(ifreq,:,:,:,:));
    [x,misfit] = mwni_irls(y, x0, T, NDim, N, K, max_iter_int, max_iter_ext, 1);
    % 用上一时刻的值
    x0 = x;
    y_recon = operator_nfft(x, ones(n1,n2,n3,n4), NDim, N, K, 1);
    Dout(ifreq,:,:,:,:) = y_recon;  % 修正: 使用 Dout 而不是 dout
  end
 end;

% Fourier domain symmetries to guarantee that time series are real when we return to time 

 if NDim == 2; 
  for k = nfft/2+2:nfft
   Dout(k,:) = conj(Dout(nfft-k+2,:));
  end
  dout = ifft(Dout,nfft,1); 
  dout = real(dout(1:nt,:)); 
 end

 if NDim == 3; 
  for k = nfft/2+2:nfft
   Dout(k,:,:) = conj(Dout(nfft-k+2,:,:));
  end
  dout = ifft(Dout,nfft,1); 
  dout = real(dout(1:nt,:,:)); 
 end

 if NDim == 4; 
  for k = nfft/2+2:nfft
   Dout(k,:,:,:) = conj(Dout(nfft-k+2,:,:,:));
  end
  dout = ifft(Dout,nfft,1); 
  dout = real(dout(1:nt,:,:,:)); 
 end

 if NDim == 5; 
  for k = nfft/2+2:nfft
   Dout(k,:,:,:,:) = conj(Dout(nfft-k+2,:,:,:,:));  % 修正: 使用 Dout 而不是 dout
  end
  dout = ifft(Dout,nfft,1);  % 修正: 使用 Dout 而不是 dout
  dout = real(dout(1:nt,:,:,:,:)); 
 end

return
