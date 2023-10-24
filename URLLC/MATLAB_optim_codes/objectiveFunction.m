%%%%%%%%%%%%%%% Defining multiple objective functions because of different
%%%%%%%%%%%%%%% constraints and different formulations and checking out which variant is better %%%%%%%%%%%%%%%%%%%%%

function cost = objectiveFunction(allocMatrix,h_prev_ts, delta, time_slot, B, U, N, k, n, Po)
    %allocMatrix = initial_allocMatrix;
    

    
    allocMatrix = descritize_alloc_matrix(allocMatrix, B, N);
    all_SINRs = zeros(B,N,U);
    for serving_BS_ind = 1:B
        all_SINRs(serving_BS_ind,:,:) = computeSINR(allocMatrix, time_slot, B, U, N, serving_BS_ind, h_prev_ts, delta);
    end

    %allocSum = sum(allocMatrix, 3);

    %SINR = calculateSINR(allocMatrix, time_slot, B, U, N);
    
    % Compute channel capacity and channel dispersion
    SINRs_per_user = sum(all_SINRs, 3);
    c = log(1 + SINRs_per_user);
    V = SINRs_per_user .* (2 + SINRs_per_user) ./ (1 + SINRs_per_user).^2 * (log(exp(1)))^2;
    
    % Compute the outage probability
    p_out = 0.5 * erfc((2*n.*c - 2*k + log(n))./(2*sqrt(n).*sqrt(V))/sqrt(2));
    
    % Objective value
    p_out_sum = sum(p_out(:));
    

    % Compute the penalty term for the constraints
    
    %penalty = sum(sum(sum(max(0, p_out - Po)))); % Only penalize values greater than Po

    % Combine the objective and the penalty
    %cost = p_out_sum + 1e5*penalty; % adjust the penalty weight (1e5) depending on the problem scale
    cost = p_out_sum;
    %disp('Objective function evaluated.');
    %fprintf('Current cost: %f\n', cost);
    % Constraint for the outage probability
    if any(p_out(:) > Po)
        %disp('Hi')
        cost = cost + 1e5; % Penalty for violation
    end

    %fprintf('cost is', cost);
    % 
    % % Constraint for the allocation matrix structure
    % allocSum = sum(allocMatrix, 3);
    % if any(allocSum(:) > 1) || any(allocSum(:) < 0)
    %     cost = cost + 1e5; % Add penalty if violated
    % end
    % 
    % % Constraint to penalize all-zero rows
    % %penalty_per_zero_row = 1e3; % Adjust based on how much to discourage all-zero rows
    % for b = 1:B
    %     subsetMatrix = squeeze(allocMatrix(b, :, :));
    %     rowSums = sum(subsetMatrix, 2);
    %     num_zero_rows = sum(rowSums == 0);
    %     cost = cost + penalty_per_zero_row * num_zero_rows;
    % end
        % % Constraint for the allocation matrix structure
    % allocSum = sum(allocMatrix, 3);
    % if any(allocSum(:) > 1) || any(allocSum(:) < 0)
    %     cost = cost + 1e5; % Add penalty if violated
    % end
    % 
    % % Constraint to penalize all-zero rows
    % %penalty_per_zero_row = 1e3; % Adjust based on how much to discourage all-zero rows
    % for b = 1:B
    %     subsetMatrix = squeeze(allocMatrix(b, :, :));
    %     rowSums = sum(subsetMatrix, 2);
    %     num_zero_rows = sum(rowSums == 0);
    %     cost = cost + penalty_per_zero_row * num_zero_rows;
    % end
  
    cost = cost_optimize_simulated_annealing(allocMatrix,h_prev_ts, delta, time_slot, B, U, N, k, n, Po, cost);
end



function cost = objectiveFunction_alternate1(allocMatrix,h_prev_ts, delta, time_slot, B, U, N, k, n, Po)
    %allocMatrix = initial_allocMatrix;
    

    
    %allocMatrix = descritize_alloc_matrix(allocMatrix, B, N);
    all_SINRs = zeros(B,N,U);
    for serving_BS_ind = 1:B
        all_SINRs(serving_BS_ind,:,:) = computeSINR(allocMatrix, time_slot, B, U, N, serving_BS_ind, h_prev_ts, delta);
    end

    %allocSum = sum(allocMatrix, 3);

    %SINR = calculateSINR(allocMatrix, time_slot, B, U, N);
    
    % Compute channel capacity and channel dispersion
    SINRs_per_user = sum(all_SINRs, 3);
    c = log(1 + SINRs_per_user);
    V = SINRs_per_user .* (2 + SINRs_per_user) ./ (1 + SINRs_per_user).^2 * (log(exp(1)))^2;
    
    % Compute the outage probability
    p_out = 0.5 * erfc((2*n.*c - 2*k + log(n))./(2*sqrt(n).*sqrt(V))/sqrt(2));
    
    % Objective value
    p_out_sum = sum(p_out(:));
    

    % Compute the penalty term for the constraints
    
    %penalty = sum(sum(sum(max(0, p_out - Po)))); % Only penalize values greater than Po

    % Combine the objective and the penalty
    %cost = p_out_sum + 1e5*penalty; % adjust the penalty weight (1e5) depending on the problem scale
    cost = p_out_sum;
    %disp('Objective function evaluated.');
    %fprintf('Current cost: %f\n', cost);
    % Constraint for the outage probability
    if any(p_out > Po)
        cost = cost + 1e5; % Penalty for violation
    end
    
    % 
    % % Constraint for the allocation matrix structure
    % allocSum = sum(allocMatrix, 3);
    % if any(allocSum(:) > 1) || any(allocSum(:) < 0)
    %     cost = cost + 1e5; % Add penalty if violated
    % end
    % 
    % % Constraint to penalize all-zero rows
    % %penalty_per_zero_row = 1e3; % Adjust based on how much to discourage all-zero rows
    % %for b = 1:B
    %     %subsetMatrix = squeeze(allocMatrix(b, :, :));
    %     %rowSums = sum(subsetMatrix, 2);
    %     %num_zero_rows = sum(rowSums == 0);
    %     %cost = cost + penalty_per_zero_row * num_zero_rows;
    % %end
end

function cost = objectiveFunction_alternate2(allocMatrix,h_prev_ts, delta, time_slot, B, U, N, k, n, Po)
    %allocMatrix = initial_allocMatrix;
    

    
    %allocMatrix = descritize_alloc_matrix(allocMatrix, B, N);
    all_SINRs = zeros(B,N,U);
    for serving_BS_ind = 1:B
        all_SINRs(serving_BS_ind,:,:) = computeSINR(allocMatrix, time_slot, B, U, N, serving_BS_ind, h_prev_ts, delta);
    end

    %allocSum = sum(allocMatrix, 3);

    %SINR = calculateSINR(allocMatrix, time_slot, B, U, N);
    
    % Compute channel capacity and channel dispersion
    SINRs_per_user = sum(all_SINRs, 3);
    c = log(1 + SINRs_per_user);
    V = SINRs_per_user .* (2 + SINRs_per_user) ./ (1 + SINRs_per_user).^2 * (log(exp(1)))^2;
    
    % Compute the outage probability
    p_out = 0.5 * erfc((2*n.*c - 2*k + log(n))./(2*sqrt(n).*sqrt(V))/sqrt(2));
    
    % Objective value
    p_out_sum = sum(p_out(:));
    

    % Compute the penalty term for the constraints
    
    %penalty = sum(sum(sum(max(0, p_out - Po)))); % Only penalize values greater than Po

    % Combine the objective and the penalty
    %cost = p_out_sum + 1e5*penalty; % adjust the penalty weight (1e5) depending on the problem scale
    cost = p_out_sum;
    %disp('Objective function evaluated.');
    %fprintf('Current cost: %f\n', cost);
    % Constraint for the outage probability
    if any(p_out > Po)
        cost = cost + 1e5; % Penalty for violation
    end
    % 
    % % Constraint for the allocation matrix structure
    % allocSum = sum(allocMatrix, 3);
    % if any(allocSum(:) > 1) || any(allocSum(:) < 0)
    %     cost = cost + 1e5; % Add penalty if violated
    % end
    % 
    % % Constraint to penalize all-zero rows
    % %penalty_per_zero_row = 1e3; % Adjust based on how much to discourage all-zero rows
    % %for b = 1:B
    %     %subsetMatrix = squeeze(allocMatrix(b, :, :));
    %     %rowSums = sum(subsetMatrix, 2);
    %     %num_zero_rows = sum(rowSums == 0);
    %     %cost = cost + penalty_per_zero_row * num_zero_rows;
    % %end
end

function cost = cost_optimize_simulated_annealing(allocMatrix,h_prev_ts, delta, time_slot, B, U, N, k, n, Po, cost)
%fraction_to_be_penalized_mean = 10^-6;
%fraction_to_be_penalized_var = 10^-8;
fraction_to_be_penalized_mean = objectiveFunction_alternate21(allocMatrix,h_prev_ts, delta, time_slot, B, U, N, k, n, Po);
fraction_to_be_penalized_var = objectiveFunction_alternate4(allocMatrix,h_prev_ts, delta, time_slot, B, U, N, k, n, Po);
cost = objectiveFunction_alternate6(allocMatrix,h_prev_ts, delta, fraction_to_be_penalized_mean, fraction_to_be_penalized_var, time_slot, B, U, N, k, n, Po);
%cost = abs(normrnd(fraction_to_be_penalized_mean, fraction_to_be_penalized_var)); 
%disp(cost)
end

function cost = objectiveFunction_alternate26(allocMatrix,h_prev_ts, delta, time_slot, B, U, N, k, n, Po)
    %allocMatrix = initial_allocMatrix;
    

    
    %allocMatrix = descritize_alloc_matrix(allocMatrix, B, N);
    all_SINRs = zeros(B,N,U);
    for serving_BS_ind = 1:B
        all_SINRs(serving_BS_ind,:,:) = computeSINR(allocMatrix, time_slot, B, U, N, serving_BS_ind, h_prev_ts, delta);
    end

    %allocSum = sum(allocMatrix, 3);

    %SINR = calculateSINR(allocMatrix, time_slot, B, U, N);
    
    % Compute channel capacity and channel dispersion
    SINRs_per_user = sum(all_SINRs, 3);
    c = log(1 + SINRs_per_user);
    V = SINRs_per_user .* (2 + SINRs_per_user) ./ (1 + SINRs_per_user).^2 * (log(exp(1)))^2;
    
    % Compute the outage probability
    p_out = 0.5 * erfc((2*n.*c - 2*k + log(n))./(2*sqrt(n).*sqrt(V))/sqrt(2));
    
    % Objective value
    p_out_sum = sum(p_out(:));
    

    % Compute the penalty term for the constraints
    
    %penalty = sum(sum(sum(max(0, p_out - Po)))); % Only penalize values greater than Po

    % Combine the objective and the penalty
    %cost = p_out_sum + 1e5*penalty; % adjust the penalty weight (1e5) depending on the problem scale
    cost = p_out_sum;
    %disp('Objective function evaluated.');
    %fprintf('Current cost: %f\n', cost);
    % Constraint for the outage probability
    if any(p_out > Po)
        cost = cost + 1e5; % Penalty for violation
    end
    % 
    % % Constraint for the allocation matrix structure
    % allocSum = sum(allocMatrix, 3);
    % if any(allocSum(:) > 1) || any(allocSum(:) < 0)
    %     cost = cost + 1e5; % Add penalty if violated
    % end
    % 
    % % Constraint to penalize all-zero rows
    % %penalty_per_zero_row = 1e3; % Adjust based on how much to discourage all-zero rows
    % %for b = 1:B
    %     %subsetMatrix = squeeze(allocMatrix(b, :, :));
    %     %rowSums = sum(subsetMatrix, 2);
    %     %num_zero_rows = sum(rowSums == 0);
    %     %cost = cost + penalty_per_zero_row * num_zero_rows;
    % %end
end


function cost = objectiveFunction_alternate24(allocMatrix,h_prev_ts, delta, time_slot, B, U, N, k, n, Po)
    %allocMatrix = initial_allocMatrix;
    

    
    %allocMatrix = descritize_alloc_matrix(allocMatrix, B, N);
    all_SINRs = zeros(B,N,U);
    for serving_BS_ind = 1:B
        all_SINRs(serving_BS_ind,:,:) = computeSINR(allocMatrix, time_slot, B, U, N, serving_BS_ind, h_prev_ts, delta);
    end

    %allocSum = sum(allocMatrix, 3);

    %SINR = calculateSINR(allocMatrix, time_slot, B, U, N);
    
    % Compute channel capacity and channel dispersion
    SINRs_per_user = sum(all_SINRs, 3);
    c = log(1 + SINRs_per_user);
    V = SINRs_per_user .* (2 + SINRs_per_user) ./ (1 + SINRs_per_user).^2 * (log(exp(1)))^2;
    
    % Compute the outage probability
    p_out = 0.5 * erfc((2*n.*c - 2*k + log(n))./(2*sqrt(n).*sqrt(V))/sqrt(2));
    
    % Objective value
    p_out_sum = sum(p_out(:));
    

    % Compute the penalty term for the constraints
    
    %penalty = sum(sum(sum(max(0, p_out - Po)))); % Only penalize values greater than Po

    % Combine the objective and the penalty
    %cost = p_out_sum + 1e5*penalty; % adjust the penalty weight (1e5) depending on the problem scale
    cost = p_out_sum;
    %disp('Objective function evaluated.');
    %fprintf('Current cost: %f\n', cost);
    % Constraint for the outage probability
    if any(p_out > Po)
        cost = cost + 1e5; % Penalty for violation
    end
    % 
    % % Constraint for the allocation matrix structure
    % allocSum = sum(allocMatrix, 3);
    % if any(allocSum(:) > 1) || any(allocSum(:) < 0)
    %     cost = cost + 1e5; % Add penalty if violated
    % end
    % 
    % % Constraint to penalize all-zero rows
    % %penalty_per_zero_row = 1e3; % Adjust based on how much to discourage all-zero rows
    % %for b = 1:B
    %     %subsetMatrix = squeeze(allocMatrix(b, :, :));
    %     %rowSums = sum(subsetMatrix, 2);
    %     %num_zero_rows = sum(rowSums == 0);
    %     %cost = cost + penalty_per_zero_row * num_zero_rows;
    % %end
end


function cost_mean = objectiveFunction_alternate21(allocMatrix,h_prev_ts, delta, time_slot, B, U, N, k, n, Po)
    %allocMatrix = initial_allocMatrix;
    

    
    %allocMatrix = descritize_alloc_matrix(allocMatrix, B, N);
    all_SINRs = zeros(B,N,U);
    cost_SINR_frac = 1e-30;
    cost_SINR_offset = 1e-6;
    for serving_BS_ind = 1:B
        all_SINRs(serving_BS_ind,:,:) = computeSINR(allocMatrix, time_slot, B, U, N, serving_BS_ind, h_prev_ts, delta);
    end

    %allocSum = sum(allocMatrix, 3);

    %SINR = calculateSINR(allocMatrix, time_slot, B, U, N);
    
    % Compute channel capacity and channel dispersion
    SINRs_per_user = sum(all_SINRs, 3);
    c = log(1 + SINRs_per_user);
    V = SINRs_per_user .* (2 + SINRs_per_user) ./ (1 + SINRs_per_user).^2 * (log(exp(1)))^2;
    
    % Compute the outage probability
    p_out = 0.5 * erfc((2*n.*c - 2*k + log(n))./(2*sqrt(n).*sqrt(V))/sqrt(2));
    
    % Objective value
    p_out_sum = sum(p_out(:));
    

    % Compute the penalty term for the constraints
    
    %penalty = sum(sum(sum(max(0, p_out - Po)))); % Only penalize values greater than Po

    % Combine the objective and the penalty
    %cost = p_out_sum + 1e5*penalty; % adjust the penalty weight (1e5) depending on the problem scale
    cost = p_out_sum;
    %disp('Objective function evaluated.');
    %fprintf('Current cost: %f\n', cost);
    % Constraint for the outage probability
    if any(p_out > Po)
        cost = cost + 1e5; % Penalty for violation
    end
    % adding constraints for the channel allocation matrix trying out
    % different penalizing constants%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%adding constraints on the variance and modelling such that SINR is 
    %%%%%%%%%%%%%%%%%%% typically in the range between -10 dB to 20 dB
    cost_mean = cost*cost_SINR_frac +cost_SINR_offset; 
    % 
    % % Constraint for the allocation matrix structure
    % allocSum = sum(allocMatrix, 3);
    % if any(allocSum(:) > 1) || any(allocSum(:) < 0)
    %     cost = cost + 1e5; % Add penalty if violated
    % end
    % 
    % % Constraint to penalize all-zero rows
    % %penalty_per_zero_row = 1e3; % Adjust based on how much to discourage all-zero rows
    % %for b = 1:B
    %     %subsetMatrix = squeeze(allocMatrix(b, :, :));
    %     %rowSums = sum(subsetMatrix, 2);
    %     %num_zero_rows = sum(rowSums == 0);
    %     %cost = cost + penalty_per_zero_row * num_zero_rows;
    % %end
end



function allocMatrix = descritize_alloc_matrix(allocMatrix, B, N)
    for b = 1:B
        A = squeeze(allocMatrix(b,:,:));
        for n = 1:N
            if any(A(n,:)>=0.5)
               [~,max_ind] = max(A(n,:));
               A(n,:) = zeros(1, numel(A(n,:)));
               A(n,max_ind) = 1.0;
            else 
                A(n,:) = zeros(1, numel(A(n,:)));
            end
        end
    allocMatrix(b,:,:) = A;
    end
end




function cost_var = objectiveFunction_alternate4(allocMatrix,h_prev_ts, delta, time_slot, B, U, N, k, n, Po)
    %allocMatrix = initial_allocMatrix;
    

    
    %allocMatrix = descritize_alloc_matrix(allocMatrix, B, N);
    all_SINRs = zeros(B,N,U);
    cost_SINR_frac = 1e-30;
    cost_SINR_offset = 1e-8;
    for serving_BS_ind = 1:B
        all_SINRs(serving_BS_ind,:,:) = computeSINR(allocMatrix, time_slot, B, U, N, serving_BS_ind, h_prev_ts, delta);
    end

    %allocSum = sum(allocMatrix, 3);

    %SINR = calculateSINR(allocMatrix, time_slot, B, U, N);
    
    % Compute channel capacity and channel dispersion
    SINRs_per_user = sum(all_SINRs, 3);
    c = log(1 + SINRs_per_user);
    V = SINRs_per_user .* (2 + SINRs_per_user) ./ (1 + SINRs_per_user).^2 * (log(exp(1)))^2;
    
    % Compute the outage probability
    p_out = 0.5 * erfc((2*n.*c - 2*k + log(n))./(2*sqrt(n).*sqrt(V))/sqrt(2));
    
    % Objective value
    p_out_sum = sum(p_out(:));
    

    % Compute the penalty term for the constraints
    
    %penalty = sum(sum(sum(max(0, p_out - Po)))); % Only penalize values greater than Po

    % Combine the objective and the penalty
    %cost = p_out_sum + 1e5*penalty; % adjust the penalty weight (1e5) depending on the problem scale
    cost = p_out_sum;
    %disp('Objective function evaluated.');
    %fprintf('Current cost: %f\n', cost);
    % Constraint for the outage probability
    if any(p_out > Po)
        cost = cost + 1e5; % Penalty for violation
    end
    %%%%%%%%% adding constraints on the variance and modelling such that SINR is typically in the range between -10 dB to 20 dB ###########
    cost_var = cost*cost_SINR_frac+cost_SINR_offset;
    % 
    % % Constraint for the allocation matrix structure
    % allocSum = sum(allocMatrix, 3);
    % if any(allocSum(:) > 1) || any(allocSum(:) < 0)
    %     cost = cost + 1e5; % Add penalty if violated
    % end
    % 
    % % Constraint to penalize all-zero rows
    % %penalty_per_zero_row = 1e3; % Adjust based on how much to discourage all-zero rows
    % %for b = 1:B
    %     %subsetMatrix = squeeze(allocMatrix(b, :, :));
    %     %rowSums = sum(subsetMatrix, 2);
    %     %num_zero_rows = sum(rowSums == 0);
    %     %cost = cost + penalty_per_zero_row * num_zero_rows;
    % %end
end




function cost = objectiveFunction_alternate5(allocMatrix,h_prev_ts, delta, time_slot, B, U, N, k, n, Po)
    %allocMatrix = initial_allocMatrix;
    

    
    %allocMatrix = descritize_alloc_matrix(allocMatrix, B, N);
    all_SINRs = zeros(B,N,U);
    for serving_BS_ind = 1:B
        all_SINRs(serving_BS_ind,:,:) = computeSINR(allocMatrix, time_slot, B, U, N, serving_BS_ind, h_prev_ts, delta);
    end

    %allocSum = sum(allocMatrix, 3);

    %SINR = calculateSINR(allocMatrix, time_slot, B, U, N);
    
    % Compute channel capacity and channel dispersion
    SINRs_per_user = sum(all_SINRs, 3);
    c = log(1 + SINRs_per_user);
    V = SINRs_per_user .* (2 + SINRs_per_user) ./ (1 + SINRs_per_user).^2 * (log(exp(1)))^2;
    
    % Compute the outage probability
    p_out = 0.5 * erfc((2*n.*c - 2*k + log(n))./(2*sqrt(n).*sqrt(V))/sqrt(2));
    
    % Objective value
    p_out_sum = sum(p_out(:));
    

    % Compute the penalty term for the constraints
    
    %penalty = sum(sum(sum(max(0, p_out - Po)))); % Only penalize values greater than Po

    % Combine the objective and the penalty
    %cost = p_out_sum + 1e5*penalty; % adjust the penalty weight (1e5) depending on the problem scale
    cost = p_out_sum;
    %disp('Objective function evaluated.');
    %fprintf('Current cost: %f\n', cost);
    % Constraint for the outage probability
    if any(p_out > Po)
        cost = cost + 1e5; % Penalty for violation
    end
    % 
    % % Constraint for the allocation matrix structure
    % allocSum = sum(allocMatrix, 3);
    % if any(allocSum(:) > 1) || any(allocSum(:) < 0)
    %     cost = cost + 1e5; % Add penalty if violated
    % end
    % 
    % % Constraint to penalize all-zero rows
    % %penalty_per_zero_row = 1e3; % Adjust based on how much to discourage all-zero rows
    % %for b = 1:B
    %     %subsetMatrix = squeeze(allocMatrix(b, :, :));
    %     %rowSums = sum(subsetMatrix, 2);
    %     %num_zero_rows = sum(rowSums == 0);
    %     %cost = cost + penalty_per_zero_row * num_zero_rows;
    % %end
end



function cost = objectiveFunction_alternate6(allocMatrix,h_prev_ts, delta, fraction_to_be_penalized_mean, fraction_to_be_penalized_var, time_slot, B, U, N, k, n, Po)
    %allocMatrix = initial_allocMatrix;
    

    
    %allocMatrix = descritize_alloc_matrix(allocMatrix, B, N);
    all_SINRs = zeros(B,N,U);
    for serving_BS_ind = 1:B
        all_SINRs(serving_BS_ind,:,:) = computeSINR(allocMatrix, time_slot, B, U, N, serving_BS_ind, h_prev_ts, delta);
    end

    %allocSum = sum(allocMatrix, 3);

    %SINR = calculateSINR(allocMatrix, time_slot, B, U, N);
    
    % Compute channel capacity and channel dispersion
    SINRs_per_user = sum(all_SINRs, 3);
    c = log(1 + SINRs_per_user);
    V = SINRs_per_user .* (2 + SINRs_per_user) ./ (1 + SINRs_per_user).^2 * (log(exp(1)))^2;
    
    % Compute the outage probability
    p_out = 0.5 * erfc((2*n.*c - 2*k + log(n))./(2*sqrt(n).*sqrt(V))/sqrt(2));
    
    % Objective value
    p_out_sum = sum(p_out(:));
    

    % Compute the penalty term for the constraints
    
    %penalty = sum(sum(sum(max(0, p_out - Po)))); % Only penalize values greater than Po

    % Combine the objective and the penalty
    %cost = p_out_sum + 1e5*penalty; % adjust the penalty weight (1e5) depending on the problem scale
    cost = p_out_sum;
    %disp('Objective function evaluated.');
    %fprintf('Current cost: %f\n', cost);
    % Constraint for the outage probability
    if any(p_out > Po)
        cost = cost + 1e5; % Penalty for violation
    end

    % 
    % % Constraint for the allocation matrix structure
    % allocSum = sum(allocMatrix, 3);
    % if any(allocSum(:) > 1) || any(allocSum(:) < 0)
    %     cost = cost + 1e5; % Add penalty if violated
    % end
    % 
    % % Constraint to penalize all-zero rows
    % %penalty_per_zero_row = 1e3; % Adjust based on how much to discourage all-zero rows
    % %for b = 1:B
    %     %subsetMatrix = squeeze(allocMatrix(b, :, :));
    %     %rowSums = sum(subsetMatrix, 2);
    %     %num_zero_rows = sum(rowSums == 0);
    %     %cost = cost + penalty_per_zero_row * num_zero_rows;
    % %end
    % Using the above cost mean and cost var from other objective functions
    %%%%%%%%%%%%%
    cost = abs(normrnd(fraction_to_be_penalized_mean, fraction_to_be_penalized_var));
end


function cost = objectiveFunction_alternate7(allocMatrix,h_prev_ts, delta, time_slot, B, U, N, k, n, Po)
    %allocMatrix = initial_allocMatrix;
    

    
    %allocMatrix = descritize_alloc_matrix(allocMatrix, B, N);
    all_SINRs = zeros(B,N,U);
    for serving_BS_ind = 1:B
        all_SINRs(serving_BS_ind,:,:) = computeSINR(allocMatrix, time_slot, B, U, N, serving_BS_ind, h_prev_ts, delta);
    end

    %allocSum = sum(allocMatrix, 3);

    %SINR = calculateSINR(allocMatrix, time_slot, B, U, N);
    
    % Compute channel capacity and channel dispersion
    SINRs_per_user = sum(all_SINRs, 3);
    c = log(1 + SINRs_per_user);
    V = SINRs_per_user .* (2 + SINRs_per_user) ./ (1 + SINRs_per_user).^2 * (log(exp(1)))^2;
    
    % Compute the outage probability
    p_out = 0.5 * erfc((2*n.*c - 2*k + log(n))./(2*sqrt(n).*sqrt(V))/sqrt(2));
    
    % Objective value
    p_out_sum = sum(p_out(:));
    

    % Compute the penalty term for the constraints
    
    %penalty = sum(sum(sum(max(0, p_out - Po)))); % Only penalize values greater than Po

    % Combine the objective and the penalty
    %cost = p_out_sum + 1e5*penalty; % adjust the penalty weight (1e5) depending on the problem scale
    cost = p_out_sum;
    %disp('Objective function evaluated.');
    %fprintf('Current cost: %f\n', cost);
    % Constraint for the outage probability
    if any(p_out > Po)
        cost = cost + 1e5; % Penalty for violation
    end
    % 
    % % Constraint for the allocation matrix structure
    % allocSum = sum(allocMatrix, 3);
    % if any(allocSum(:) > 1) || any(allocSum(:) < 0)
    %     cost = cost + 1e5; % Add penalty if violated
    % end
    % 
    % % Constraint to penalize all-zero rows
    % %penalty_per_zero_row = 1e3; % Adjust based on how much to discourage all-zero rows
    % %for b = 1:B
    %     %subsetMatrix = squeeze(allocMatrix(b, :, :));
    %     %rowSums = sum(subsetMatrix, 2);
    %     %num_zero_rows = sum(rowSums == 0);
    %     %cost = cost + penalty_per_zero_row * num_zero_rows;
    % %end
end


function cost = objectiveFunction_alternate8(allocMatrix,h_prev_ts, delta, time_slot, B, U, N, k, n, Po)
    %allocMatrix = initial_allocMatrix;
    

    
    %allocMatrix = descritize_alloc_matrix(allocMatrix, B, N);
    all_SINRs = zeros(B,N,U);
    for serving_BS_ind = 1:B
        all_SINRs(serving_BS_ind,:,:) = computeSINR(allocMatrix, time_slot, B, U, N, serving_BS_ind, h_prev_ts, delta);
    end

    %allocSum = sum(allocMatrix, 3);

    %SINR = calculateSINR(allocMatrix, time_slot, B, U, N);
    
    % Compute channel capacity and channel dispersion
    SINRs_per_user = sum(all_SINRs, 3);
    c = log(1 + SINRs_per_user);
    V = SINRs_per_user .* (2 + SINRs_per_user) ./ (1 + SINRs_per_user).^2 * (log(exp(1)))^2;
    
    % Compute the outage probability
    p_out = 0.5 * erfc((2*n.*c - 2*k + log(n))./(2*sqrt(n).*sqrt(V))/sqrt(2));
    
    % Objective value
    p_out_sum = sum(p_out(:));
    

    % Compute the penalty term for the constraints
    
    %penalty = sum(sum(sum(max(0, p_out - Po)))); % Only penalize values greater than Po

    % Combine the objective and the penalty
    %cost = p_out_sum + 1e5*penalty; % adjust the penalty weight (1e5) depending on the problem scale
    cost = p_out_sum;
    %disp('Objective function evaluated.');
    %fprintf('Current cost: %f\n', cost);
    % Constraint for the outage probability
    if any(p_out > Po)
        cost = cost + 1e5; % Penalty for violation
    end
    % 
    % % Constraint for the allocation matrix structure
    % allocSum = sum(allocMatrix, 3);
    % if any(allocSum(:) > 1) || any(allocSum(:) < 0)
    %     cost = cost + 1e5; % Add penalty if violated
    % end
    % 
    % % Constraint to penalize all-zero rows
    % %penalty_per_zero_row = 1e3; % Adjust based on how much to discourage all-zero rows
    % %for b = 1:B
    %     %subsetMatrix = squeeze(allocMatrix(b, :, :));
    %     %rowSums = sum(subsetMatrix, 2);
    %     %num_zero_rows = sum(rowSums == 0);
    %     %cost = cost + penalty_per_zero_row * num_zero_rows;
    % %end
end


function SINR = computeSINR(allocMatrix, time_slot, B, U, N, serving_BS_ind, h_prev_ts, delta)

    %h_prev_ts = h_init;
    %serving_BS_ind = 1;

    rho_0 = 0.8;
    mu_0 = sqrt(1-rho_0^2);
    %h_present_ts = rho_0*h_prev_ts + mu_0*sqrt(0.5)*(randn(B,N,U) + 1j*randn(B,N,U));
    h_present_ts = rho_0*h_prev_ts + mu_0*delta;
    %h = sqrt(0.5)*(randn(N,U) + 1j*randn(N,U));
    %h_other = sqrt(0.5)*(randn(B-1,N,U) + 1j*randn(B-1,N,U));
    % get the indices of non serving BSs 
    non_serving_BS_ind = setdiff(1:B, serving_BS_ind);
    int_BS_allo_mat = allocMatrix(non_serving_BS_ind,:,:);
    int_BS_chs = h_present_ts(non_serving_BS_ind,:,:);
    serv_BS_chs = h_present_ts(serving_BS_ind,:,:);
    SINR =  abs(serv_BS_chs).^2./(abs(int_BS_chs).^2.*int_BS_allo_mat);
    SINR(isinf(SINR)) = 0;
    %disp('end');
end



function cost = objectiveFunction_alternate9(allocMatrix,h_prev_ts, delta, time_slot, B, U, N, k, n, Po)
    %allocMatrix = initial_allocMatrix;
    

    
    %allocMatrix = descritize_alloc_matrix(allocMatrix, B, N);
    all_SINRs = zeros(B,N,U);
    for serving_BS_ind = 1:B
        all_SINRs(serving_BS_ind,:,:) = computeSINR(allocMatrix, time_slot, B, U, N, serving_BS_ind, h_prev_ts, delta);
    end

    %allocSum = sum(allocMatrix, 3);

    %SINR = calculateSINR(allocMatrix, time_slot, B, U, N);
    
    % Compute channel capacity and channel dispersion
    SINRs_per_user = sum(all_SINRs, 3);
    c = log(1 + SINRs_per_user);
    V = SINRs_per_user .* (2 + SINRs_per_user) ./ (1 + SINRs_per_user).^2 * (log(exp(1)))^2;
    
    % Compute the outage probability
    p_out = 0.5 * erfc((2*n.*c - 2*k + log(n))./(2*sqrt(n).*sqrt(V))/sqrt(2));
    
    % Objective value
    p_out_sum = sum(p_out(:));
    

    % Compute the penalty term for the constraints
    
    %penalty = sum(sum(sum(max(0, p_out - Po)))); % Only penalize values greater than Po

    % Combine the objective and the penalty
    %cost = p_out_sum + 1e5*penalty; % adjust the penalty weight (1e5) depending on the problem scale
    cost = p_out_sum;
    %disp('Objective function evaluated.');
    %fprintf('Current cost: %f\n', cost);
    % Constraint for the outage probability
    if any(p_out > Po)
        cost = cost + 1e5; % Penalty for violation
    end
    % 
    % % Constraint for the allocation matrix structure
    % allocSum = sum(allocMatrix, 3);
    % if any(allocSum(:) > 1) || any(allocSum(:) < 0)
    %     cost = cost + 1e5; % Add penalty if violated
    % end
    % 
    % % Constraint to penalize all-zero rows
    % %penalty_per_zero_row = 1e3; % Adjust based on how much to discourage all-zero rows
    % %for b = 1:B
    %     %subsetMatrix = squeeze(allocMatrix(b, :, :));
    %     %rowSums = sum(subsetMatrix, 2);
    %     %num_zero_rows = sum(rowSums == 0);
    %     %cost = cost + penalty_per_zero_row * num_zero_rows;
    % %end
end


