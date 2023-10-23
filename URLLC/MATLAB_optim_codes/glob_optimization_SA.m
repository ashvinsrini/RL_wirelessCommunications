%%%%%%%%%%%% Define global variables %%%%%%%%%%%

N = 100;
B = 2;
U = 4;
time_slot = 1;
k = 200;
n = 500;
Po = 10^-5;
% Initialize the allocation matrix A
%initial_allocMatrix = zeros(B, N, U);
h_init= sqrt(0.5)*(randn(B,N,U) + 1j*randn(B,N,U));
delta = sqrt(0.5)*(randn(B,N,U) + 1j*randn(B,N,U));

% for b = 1:B
%     for n = 1:N
%         user_pool = 1:U;
%         u = randsample(user_pool, 1); % Randomly pick a user from the pool
% initial_allocMatrix(b, n, u) = 1;
%     end
% end

initial_allocMatrix = zeros(B, N, U);
b = 1;
    for n = 1:N
        user_pool = 1:U;
        u = randsample(user_pool, 1); % Randomly pick a user from the pool
initial_allocMatrix(b, n, u) = 1;
    end


b = 2;
    for n = 1:N
        user_pool = 1:U;
        u = randsample(user_pool, 1); % Randomly pick a user from the pool
initial_allocMatrix(b, n, u) = 1;
    end

%seed = 12345; % You can choose any positive integer here
%rng(seed);
%initial_allocMatrix = rand(B, N, U);
%%%%%%%%%% Generate SINR for every time slot %%%%%%%%%%%%%%
%%%%%%%% Global SINR has dimension B x N x U %%%%%%%%%%%%%%
% all_SINRs = zeros(B,N,U);
% for serving_BS_ind = 1:B
%     all_SINRs(serving_BS_ind,:,:) = computeSINR(initial_allocMatrix, time_slot, B, U, N, serving_BS_ind, h_init);
% end


%serving_BS_ind = 1;
%SINR = computeSINR(initial_allocMatrix, time_slot, B, U, N, serving_BS_ind, h_init);

% figure;
% SINR = all_SINRs(1,:,:);
% SINR_vect = SINR(:);
% SINR_vect = SINR_vect(SINR_vect ~= 0);
% 
% histogram(10 * log10(SINR_vect), 50);
% hold on;
% SINR = all_SINRs(2,:,:);
% SINR_vect = SINR(:);
% SINR_vect = SINR_vect(SINR_vect ~= 0);
% histogram(10 * log10(SINR_vect), 50);
% hold off;



%%%%%%%%%%%% function to define cost function 

num_iterations = 100;
optimized_outage_prob = zeros(num_iterations,1);
for time_slot = 1:num_iterations
    % Simulated Annealing
    options = optimoptions('simulannealbnd','MaxIterations',1000,'MaxFunctionEvaluations',50000);
    lb = zeros(B, N, U);
    ub = ones(B, N, U);
    %lb = [];
    %ub = [];
    [optimized_allocMatrix, fval] = simulannealbnd(@(allocMatrix) objectiveFunction(allocMatrix,h_init, delta, time_slot, B, U, N, k, n, Po), initial_allocMatrix, lb, ub, options);
    
    % Ensure optimized_allocMatrix is binary after optimization
    %optimized_allocMatrix = descritize_alloc_matrix(optimized_allocMatrix, B, N);
    
    %display(fval);
    optimized_outage_prob(time_slot) = fval;
    fprintf('time-slot: %f\n', time_slot);
    fprintf('lowest block error prob upon convergence: %.20f\n', fval);
end


optimized_allocMatrix = descritize_alloc_matrix(optimized_allocMatrix, B, N);
plot(optimized_outage_prob)

save('optimized_outage_prob.mat', 'optimized_outage_prob');








