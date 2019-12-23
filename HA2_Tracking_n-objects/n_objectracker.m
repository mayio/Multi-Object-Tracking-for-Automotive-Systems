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
    end
end

