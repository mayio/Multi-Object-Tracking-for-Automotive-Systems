classdef n_objectracker
    %N_OBJECTRACKER is a class containing functions to track n object in
    %clutter. 
    %Model structures need to be called:
    %sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time
    %           scan, Poisson distributed --- scalar 
    %           pdf_c: clutter (Poisson) intensity --- scalar
    %           intensity_c: clutter (Poisson) intensity --- scalar
    %motionmodel: a structure specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %measmodel: a structure specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the object
    %           state 
    %           R: measurement noise covariance matrix
    
    properties
        gating      %specify gating parameter
        reduction   %specify hypothesis reduction parameter
        density     %density class handle
    end
    
    methods
        
        function obj = initialize(obj,density_class_handle,P_G,m_d,w_min,merging_threshold,M)
            %INITIATOR initializes n_objectracker class
            %INPUT: density_class_handle: density class handle
            %       P_D: object detection probability
            %       P_G: gating size in decimal --- scalar
            %       m_d: measurement dimension --- scalar
            %       wmin: allowed minimum hypothesis weight --- scalar
            %       merging_threshold: merging threshold --- scalar
            %       M: allowed maximum number of hypotheses --- scalar
            %OUTPUT:  obj.density: density class handle
            %         obj.gating.P_G: gating size in decimal --- scalar
            %         obj.gating.size: gating size --- scalar
            %         obj.reduction.w_min: allowed minimum hypothesis
            %         weight in logarithmic scale --- scalar 
            %         obj.reduction.merging_threshold: merging threshold
            %         --- scalar 
            %         obj.reduction.M: allowed maximum number of hypotheses
            %         used in TOMHT --- scalar 
            obj.density = density_class_handle;
            obj.gating.P_G = P_G;
            obj.gating.size = chi2inv(obj.gating.P_G,m_d);
            obj.reduction.w_min = log(w_min);
            obj.reduction.merging_threshold = merging_threshold;
            obj.reduction.M = M;
        end
        
        function S = computeInnovationCovariance(obj, state, measmodel)
            %Measurement model Jacobian
            Hx = measmodel.H(state.x);
            %Innovation covariance
            S = Hx * state.P * Hx' + measmodel.R;
            %Make sure matrix S is positive definite
            S = (S+S') / 2;     
        end
        
        function estimates = GNNfilter(obj, states, Z, sensormodel, motionmodel, measmodel)
            %GNNFILTER tracks n object using global nearest neighbor
            %association 
            %INPUT: obj: an instantiation of n_objectracker class
            %       states: structure array of size (1, number of objects)
            %       with two fields: 
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of size (measurement
            %       dimension) x (number of measurements at corresponding
            %       time step)  
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x (number of objects)
            
            % number of states (objects)
            n_states = numel(states);
            estimates = cell(n_states,1);
            l_0_log = log(1 - sensormodel.P_D);
            l_clut_log = log(sensormodel.P_D/sensormodel.intensity_c);
            
            % for all time steps
            for k=1:numel(Z)
                z = Z{k};
                
                % number of measurements
                mk = size(z, 2);

                % cost matrix
                % set all elements of the cost matrix to infinity
                % as we set the cost only for measurement object pairs that
                % are within the gate
                L = Inf([n_states mk + n_states]);

                % for all states
                for i=1:n_states
                    state = states(i);
                    S_i = obj.computeInnovationCovariance(state, measmodel);
                    l_i_log = l_clut_log - (1/2) * log(det(2 * pi * S_i));
                    z_hat_i = measmodel.h(state.x);
                    log_likelihood = @(z) -(l_i_log - ...
                        (1/2) * (z - z_hat_i)' / S_i * (z - z_hat_i));
                    
                    % gating
                    [z_ingate, z_mask] = obj.density.ellipsoidalGating(...
                        state, z, measmodel, obj.gating.size);
                    
                    % calculate the log-likelihood for each measurement
                    mk_in_gate = size(z_ingate, 2);
                    z_idx = find(z_mask);
                    
                    for j=1:mk_in_gate
                        L(i, z_idx(j)) = log_likelihood(z_ingate(:,j));
                    end
                    
                    L(i, mk + i) = l_0_log;
                end
                
                % filter columns in the cost matrix with only infinite 
                % entries
                noninf_col_idx = sum(isinf(L)) < n_states;
                L = L(:, noninf_col_idx);
                z = z(:, noninf_col_idx(1:mk));                

                % number of remaining measurements
                mk = size(z, 2);
                
                % find the best assignment matrix using a 2D assignment
                % solver
                % col4row: A numRowX1 vector where the entry in each 
                %   element is an assignment of the element in that row to
                %   a column. 0 entries signify unassigned rows. If the 
                %   problem is infeasible, this is an empty matrix.
                [col4row,~,gain] = assign2D(L);
                assert(gain ~= -1, ...
                    'Assignment problem is not possible to solve');
                
                % create new local hypotheses according to the best 
                % assignment matrix obtained;
                % for all states
                estimates{k} = zeros(size(states(1).x,1), n_states);
                
                for i=1:n_states
                    state = states(i);
                    
                    % if object i was assigned to a measurment, 
                    % do a Kalman update
                    if col4row(i) <= mk
                        state = obj.density.update(...
                            state, z(:,col4row(i)), measmodel);
                    end
                    
                    % extract object state estimates;
                    estimates{k}(:,i) = state.x;
                    
                    % predict each local hypothesis.
                    state = obj.density.predict(state,motionmodel);
                    states(i) = state;
                end  
            end
        end
        
        function estimates = JPDAfilter(obj, states, Z, sensormodel, motionmodel, measmodel)
            %JPDAFILTER tracks n object using joint probabilistic data
            %association
            %INPUT: obj: an instantiation of n_objectracker class
            %       states: structure array of size (1, number of objects)
            %       with two fields: 
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of size (measurement
            %       dimension) x (number of measurements at corresponding
            %       time step)  
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x (number of objects)
            
            %{
                1. implement ellipsoidal gating for each local hypothesis seperately;
                2. construct 2D cost matrix of size (number of objects, number of measurements that at least fall inside the gates + number of objects);
                3. find the M best assignment matrices using a M-best 2D assignment solver;
                4. normalise the weights of different data association hypotheses;
                5. prune assignment matrices that correspond to data association hypotheses with low weights and renormalise the weights;
                6. create new local hypotheses for each of the data association results;
                7. merge local hypotheses that correspond to the same object by moment matching;
                8. extract object state estimates;
                9. predict each local hypothesis.
            %}                
                
            % number of states (objects)
            n_states = numel(states);
            estimates = cell(numel(Z),1);
            l_0_log = log(1 - sensormodel.P_D);
            l_clut_log = log(sensormodel.P_D/sensormodel.intensity_c);
            
            % number of hypotheses
            M = obj.reduction.M;
            
            % for all time steps
            for k=1:numel(Z)
                z = Z{k};
                
                % number of measurements
                mk = size(z, 2);

                % cost matrix
                % set all elements of the cost matrix to infinity
                % as we set the cost only for measurement object pairs that
                % are within the gate
                %
                % rows correspond to objects, 
                % wheras columns correspond to measurements
                L = Inf([n_states mk + n_states]);

                % for all states
                for i=1:n_states
                    state = states(i);
                    
                    % gating
                    [z_ingate, z_mask] = obj.density.ellipsoidalGating(...
                        state, z, measmodel, obj.gating.size);
                    
                    mk_in_gate = size(z_ingate, 2);
                    z_idx = find(z_mask);
                    
                    % calculate the log-likelihood for each measurement
                    % in the gate
                    S_i = obj.computeInnovationCovariance(state, measmodel);
                    l_i_log = l_clut_log - (1/2) * log(det(2 * pi * S_i));
                    z_hat_i = measmodel.h(state.x);
                    log_likelihood = @(z) -(l_i_log - ...
                        (1/2) * (z - z_hat_i)' * inv(S_i) * (z - z_hat_i));
                    
                    for j=1:mk_in_gate
                        L(i, z_idx(j)) = log_likelihood(z_ingate(:,j));
                    end
                    
                    % Log-likelihood of misdetection
                    L(i, mk + i) = -l_0_log;
                end
                
                % filter columns in the cost matrix with 
                % only infinite entries
                noninf_col_idx = sum(isinf(L)) < n_states;
                L = L(:, noninf_col_idx);
                z = z(:, noninf_col_idx(1:mk));                

                % number of remaining measurements
                mk = size(z, 2);
                
                % find the best assignment matrix using the
                % M-best 2D assignment solver
                %
                % col4rowBest: A numRowXk vector where the entry in each 
                %    element is an assignment of the element in that row 
                %    to a column. 0 entries signify unassigned rows.
                %    The columns of this matrix are the hypotheses.
                [col4rowBest,~,gainBest] = kBest2DAssign(L, M);
                assert(all(gainBest ~= -1), ...
                    'Assignment problem is not possible to solve');
                
                % there might be not as many hypotheses available
                M_left = length(gainBest);
                
                % compute the weight for each remaining hypotheses
                % The weight is exp(-tr(A' * L) but we are intereseted in
                % log weights. 
                h = 1:M_left; % hypotheses indexes
                trace_AL_h = @(h) sum(...
                    L(...
                        sub2ind(size(L), 1:n_states, col4rowBest(:,h)')));
                w_log_h = arrayfun(trace_AL_h, h) * -1;
                
                % set the misdetection hypotheses to the last index
                % as they are all the same and can be represended with 
                % one value.
                col4rowBest( col4rowBest > mk ) = mk+1;
                
                %normalize log weights
                w_log_h = normalizeLogWeights(w_log_h);
                
                % prune assignment matrices that correspond to data 
                % association hypotheses with low weights and renormalise 
                % the weights
                [w_log_h, h] = hypothesisReduction.prune(...
                    w_log_h, h, obj.reduction.w_min );
                col4rowBest = col4rowBest(:, h);
                w_log_h = normalizeLogWeights(w_log_h);
                
                
                % create new local hypotheses for each of the data 
                % association results
                %
                % compute the marginal probabilities that an object
                % was associated to measurement j. The index j = mk + 1
                % denotes the marginal probability of misdetection;
                beta_ij = zeros(n_states, mk + 1);
                n_theta = size(w_log_h, 2);
                
                for i=1:n_states
                    for h = 1:n_theta
                        j = col4rowBest(i, h);
                        beta_ij(i, j) = beta_ij(i, j) + exp(w_log_h(h));
                    end
                end
                
                % merge local hypotheses that correspond to the same object
                % by moment matching
                
                for i=1:n_states
                    state = states(i);
                    Hx = measmodel.H(state.x);
                    S = obj.computeInnovationCovariance(state, measmodel);
                    K = state.P * Hx' / S;
                    
                    dim = size(z,1);
                    eps = zeros(dim, 1);
                    eps_ij_sq_sum = zeros(dim, dim);
                    
                    if (mk > 0)
                        % update mean
                        z_hat = measmodel.h(state.x);
                        eps_ij = z - z_hat;
                        eps = sum(beta_ij(i,1:mk) .* eps_ij, 2);
                        state.x = state.x + K * eps;
                        
                        % update covariance
                        eps_ij_cell = mat2cell(...
                            eps_ij, size(eps_ij,1),ones(1, size(eps_ij,2)))';                    

                        eps_ij_weighted_squared = arrayfun(...
                            @(j) beta_ij(i,j) *  eps_ij_cell{j} * eps_ij_cell{j}', 1:size(z,2), 'UniformOutput', false)';
                        if (length(eps_ij_weighted_squared) > 1)
                            eps_ij_sq_sum = sum(cat(dim + 1,eps_ij_weighted_squared{:}),dim + 1);
                        else
                            eps_ij_sq_sum = eps_ij_weighted_squared{1};
                        end
                    end
                                        
                    P_bar = state.P - K * S * K';
                    P_tilde = K * (eps_ij_sq_sum - eps * eps') * K';
                    beta0 = beta_ij(i, mk + 1);
                    state.P = beta0 * state.P + (1 - beta0) * P_bar + P_tilde;
                
                    % extract object state estimates;
                    estimates{k}(:,i) = state.x;
                    
                    % predict each local hypothesis.
                    state = obj.density.predict(state,motionmodel);
                    states(i) = state;
                end
            end
        end
        
        function estimates = TOMHT(obj, states, Z, sensormodel, motionmodel, measmodel)
            %TOMHT tracks n object using track-oriented multi-hypothesis tracking
            %INPUT: obj: an instantiation of n_objectracker class
            %       states: structure array of size (1, number of objects)
            %       with two fields: 
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of size (measurement
            %       dimension) x (number of measurements at corresponding
            %       time step)  
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x (number of objects)
            
            % Task description
            %
            % for each local hypothesis in each hypothesis tree: 
            % 1). implement ellipsoidal gating; 
            % 2). calculate missed detection and predicted likelihood for
            %     each measurement inside the gate and make sure to save
            %     these for future use; 
            % 3). create updated local hypotheses and make sure to save how
            %     these connects to the old hypotheses and to the new the 
            %     measurements for future use;
            %
            % for each predicted global hypothesis: 
            % 1). create 2D cost matrix; 
            % 2). obtain M best assignments using a provided M-best 2D 
            %     assignment solver; 
            % 3). update global hypothesis look-up table according to 
            %     the M best assignment matrices obtained and use your new 
            %     local hypotheses indexing;
            %
            % normalise global hypothesis weights and implement hypothesis
            % reduction technique: pruning and capping;
            %
            % prune local hypotheses that are not included in any of the
            % global hypotheses;
            %
            % Re-index global hypothesis look-up table;
            %
            % extract object state estimates from the global hypothesis
            % with the highest weight;
            %
            % predict each local hypothesis in each hypothesis tree.
            %
            %
            
            % number of states (objects)
            n_states = numel(states);
            estimates = cell(numel(Z),1);
            l_0_log = log(1 - sensormodel.P_D);
            l_clut_log = log(sensormodel.P_D/sensormodel.intensity_c);
            
            % number of hypotheses
            M = obj.reduction.M;
            
            % create initial hypotheses trees
            local_h_trees = cell(1,n_states);
            new_local_h_trees = cell(1,n_states);
            
            for i=1:n_states
                local_h_trees{i} = states(i);
            end
            
            % initial log weights for each state
            w_log_states = cell(1, n_states)
            
            % initialize the global hypotheses lookup table
            % with one hypothesis (local hypothesis) for each state.
            % For each object there is only one local hypothesis in the 
            % beginning. Thus index 1.
            %
            % rows are global hypotheses
            % columns are states
            global_H = ones(1, n_states);
            
            % there is only one hypothesis, so the hypotheses 
            % weight is only 1
            w_log_h = log(1);
            
            % for all time steps
            for k=1:numel(Z)
                z = Z{k};
                
                % number of measurements
                mk = size(z, 2);

                % there is a gating for each state / hypothesis tree
                % each cell contains a matrix
                %    measurements x local hypothesis (each column is a
                %    leaf)
                %       
                z_masks = cell(1, n_states);

                % for each local hypothesis tree (object)
                for i=1:n_states
                    % number of leafs / number of local hypotheses
                    n_lh = length(local_h_trees{i});                    

                    % get leafs (local hypotheses)
                    lhs = local_h_trees{i};
                                        
                    % for each leaf (local hypothesis) do gating
                    z_masks{i} = zeros(mk,n_lh);
                    
                    for i_lh = 1:n_lh
                        % gating
                        [~, z_masks{i}(:,i_lh)] = obj.density.ellipsoidalGating(...
                            lhs(i_lh), z, measmodel, obj.gating.size);
                    end
                end
                
                % get rid of all measurements that were 
                % not associated at all
                non_associated_z = ...
                    sum(cell2mat(z_masks)') == 0;
                z = z(:,~non_associated_z);
                mk = sum(~non_associated_z);
                z_masks = cellfun(...
                    @(z_masks_i) z_masks_i(~non_associated_z,:), ...
                    z_masks, 'UniformOutput' ,false);

                % calculate missed detection and predicted likelihood for
                % each measurement inside the gate and save
                % these for future use; 

                % for each local hypothesis tree (object)
                for i=1:n_states
                    % number of leafs / number of local hypotheses
                    n_lh = length(local_h_trees{i});                    

                    % initialize log weights with infinity
                    w_log_states{i} = -inf(1, n_lh * (mk + 1));

                    % get leafs (local hypotheses)
                    lhs = local_h_trees{i};
                                        
                    for lh_i = 1:n_lh
                        lh = lhs(lh_i);
                        
                        % calculate missed detection and predicted
                        % likelihood
                        S = obj.computeInnovationCovariance(lh, measmodel);                        
                        z_bar = measmodel.h(lh.x);
                        
                        for j = find(z_masks{i}(:, lh_i))'
                            % new local hypothesis index
                            new_lh_i = (lh_i - 1) * (mk + 1) + j;
                            
                            w_log_states{i}(new_lh_i) = l_clut_log -...
                                1/2 * log(det( 2 * pi * S)) - ...
                                1/2 * (z(:, j) - z_bar).' / S * (z(:, j) - z_bar);
                            
                            % create updated local hypotheses
                            new_local_h_trees{i}(new_lh_i) = obj.density.update(...
                                lh, z(:,j) , measmodel);                            
                        end
                        
                        misdetect_lh_i = lh_i * (mk + 1);
                        
                        % calculate missed detection likelihood
                        w_log_states{i}(misdetect_lh_i) = l_0_log;
                        
                        % create updated misdetection local hypotheses
                        new_local_h_trees{i}(misdetect_lh_i) = lhs(lh_i);
                    end
                end
                
                % for each predicted global hypothesis: 
                % (old hypothesis trees)
                new_global_H = [];
                new_w_log_h = [];
                
                for H_i = 1:size(global_H, 1);
                    % 1). create 2D cost matrix; 
                    % 
                    % number of states rows, number of measurements + number
                    % of misdetection columns
                    L = inf(n_states, mk + n_states);

                    for i=1:n_states
                        local_h_i = global_H(H_i, i);
                        first_i = (local_h_i - 1) * (mk + 1) + 1;
                        last_i  =  local_h_i      * (mk + 1) - 1;
                        L(i, 1:mk) = -w_log_states{i}(first_i:last_i);
                        
                        % misdetection hypthesis
                        L(i, mk + i) = -w_log_states{i}(last_i + 1);
                    end

                    % 2). obtain M best assignments using a provided  
                    %     M-best 2D assignment solver; 
                    
                    % col4rowBest: A numRowXk vector where the entry in each 
                    %    element is an assignment of the element in that row 
                    %    to a column. 0 entries signify unassigned rows.
                    %    The columns of this matrix are the hypotheses.
                    [col4rowBest,~,gainBest] = kBest2DAssign(L, M);
                    assert(all(gainBest ~= -1), ...
                        'Assignment problem is not possible to solve');

                    % there might be not as many hypotheses available
                    M_left = length(gainBest);

                    % compute the weight for each remaining hypotheses
                    % The weight is exp(-tr(A' * L) but we are intereseted in
                    % log weights. 
                    M_i = 1:M_left; % hypotheses indexes
                    trace_AL_h = @(h) sum(...
                        L(...
                            sub2ind(size(L), 1:n_states, col4rowBest(:,h)')));
                    new_w_log_h_i = arrayfun(trace_AL_h, M_i) * -1;
                    new_w_log_h_i = new_w_log_h_i + w_log_h(H_i);
                    new_w_log_h = [new_w_log_h;new_w_log_h_i'];

                    % set the misdetection hypotheses to the last index
                    % as they are all the same and can be represended with 
                    % one value.                    
                    col4rowBest( col4rowBest > mk ) = mk+1;
                    
                    % 3). update global hypothesis look-up table according 
                    %     to the M best assignment matrices obtained and 
                    %     use your new local hypotheses indexing;
                    for M_i=1:M_left
                        % we use the prior look-up-table, the prior
                        % hypothesis and the data association to figure out
                        % which posterior local hypothesis is included in
                        % the look-up-table
                        new_global_H(end + 1, :) = ...
                            arrayfun(...
                                @(i) (global_H(H_i, i) - 1) * (mk + 1) + col4rowBest(i, M_i), ...
                                1:n_states);
                    end
                end

                %normalize log weights
                new_w_log_h = normalizeLogWeights(new_w_log_h);

                % prune assignment matrices that correspond to data 
                % association hypotheses with low weights and renormalise 
                % the weights
                [new_w_log_h, hyp_left] = hypothesisReduction.prune(...
                    new_w_log_h, 1:length(new_w_log_h), obj.reduction.w_min );
                new_global_H = new_global_H(hyp_left, :);

                %normalize log weights
                new_w_log_h = normalizeLogWeights(new_w_log_h);
                
                % capping
                [new_w_log_h, hyp_left] = hypothesisReduction.cap(...
                    new_w_log_h, 1:length(new_w_log_h), obj.reduction.M );
                new_global_H = new_global_H(hyp_left, :);
                
                %normalize log weights
                new_w_log_h = normalizeLogWeights(new_w_log_h);
                    
                for i=1:n_states                    
                    % prune local hypotheses that are not included in any
                    % of the global hypotheses;
                    hyp_keep = unique(new_global_H(:, i));
                    new_local_h_trees{i} = new_local_h_trees{i}(hyp_keep);
                    
                    % Re-index global hypothesis look-up table;
                    for new_index_i=1:numel(hyp_keep)
                        new_global_H( ...
                            new_global_H(:,i) == hyp_keep(new_index_i), i)... 
                            = new_index_i;
                    end
                end

                
                [~,best_w_log_h_idx] = max(new_w_log_h);
                
                for i=1:n_states
                    % extract object state estimates from the global hypothesis
                    % with the highest weight;
                    estimates{k}(:,i)   = new_local_h_trees{i}( ...
                        new_global_H(best_w_log_h_idx, i)).x;
                    
                    % predict each local hypothesis in each hypothesis 
                    % tree.
                    new_local_h_trees{i} = arrayfun(...
                        @(lh) obj.density.predict(lh, motionmodel), ...
                        new_local_h_trees{i});
                end
                
                local_h_trees = new_local_h_trees;
                w_log_h = new_w_log_h;
                global_H = new_global_H;
            end    
        end
    end
end

