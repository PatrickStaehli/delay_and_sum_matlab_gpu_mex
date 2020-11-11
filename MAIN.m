% 
% This example demonstrates how to implement and run a CUDA implementation of a conventiona delay-and-sum
% algorithm for plane-wave US acquisitions with various different steering angles. 
%
% author: Patrick Stähli
% last update: 11.11.2020


clear
%% load signal data
disp('Loading data...');
load('Example_data');

% The RF Signal and the corresponding acquisition parameters are defined in Example_data

%% Define parameters and properties

recon_parameter.c                                   = 1.50;                                             % Anticipated SoS for image reconstruction mm/us
recon_parameter.pitch                               = acquisition_parameter.pitch;                      % element distance
recon_parameter.dx                                  = recon_parameter.pitch;                            % [mm] pixel size in x
recon_parameter.dz                                  = 0.5*recon_parameter.c/acquisition_parameter.F;    % resolution in z direction
recon_parameter.Nx                                  = (acquisition_parameter.N_piezos-1);               % number of pixels in x
recon_parameter.aperture                            = (recon_parameter.Nx-1)*recon_parameter.dx;        % [mm]
recon_parameter.N_angles                            = acquisition_parameter.N_angles;
recon_parameter.tx_angles                           = acquisition_parameter.tx_angles;                  % tx abgkes
recon_parameter.range                               = 50;                                               % [mm]  Range for the reconstruction
recon_parameter.apodtan                             = tand(35);                                         % Element apod tan
recon_parameter.Nz                                  = round(recon_parameter.range/recon_parameter.dz) + 1;
recon_parameter.number_of_angles                    = acquisition_parameter.N_angles;                   % Number of reconstructed angles
recon_parameter.lense.thickness                     = 0.812;                                            % [mm] acoustic lens thickness
recon_parameter.lense.c                             = 1.015;                                            % [mm] acoustic lens thickness

%% Delay and sum
tic;
tx_angles = asind(sind(acquisition_parameter.tx_angles)*recon_parameter.c/acquisition_parameter.c0);

% Generate GPU arrays
tx_delays_GPU   = gpuArray(acquisition_parameter.tx_delays);
tx_angles_GPU   = gpuArray(tx_angles*pi/180);
raw_data_GPU    = gpuArray(double(real(raw_data(1:recon_parameter.Nz,:,:))));

% Call the mex function
pw_data_GPU = delay_and_sum_GPU_real(raw_data_GPU,acquisition_parameter,recon_parameter, tx_delays_GPU, tx_angles_GPU);

% Transfer form device to host
pw_data = gather(pw_data_GPU);

disp(['Elapsed time for delay and sum: ' , num2str(toc), ' seconds'])
%% Display the image
dBrange = 30;
for rx_ind = 1:recon_parameter.N_angles
    sa_bmode(:,:,rx_ind) = 10*log10(abs(pw_data(:,:,rx_ind)).^2) - 60;
end
imagesc(mean(sa_bmode,3),[0 dBrange]);
colormap gray
