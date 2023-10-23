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