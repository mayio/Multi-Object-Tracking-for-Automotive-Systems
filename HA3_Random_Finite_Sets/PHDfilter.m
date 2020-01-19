classdef PHDfilter
    %PHDFILTER is a class containing necessary functions to implement the
    %PHD filter 
    %Model structures need to be called:
    %    sensormodel: a structure which specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time scan, 
    %                     Poisson distributed --- scalar
    %           pdf_c: value of clutter pdf --- scalar
    %           intensity_c: Poisson clutter intensity --- scalar
    %       motionmodel: a structure which specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %       measmodel: a structure which specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the target state
    %           R: measurement noise covariance matrix
    %       birthmodel: a structure array which specifies the birth model (Gaussian
    %       mixture density) parameters --- (1 x number of birth components)
    %           w: weights of mixture components (in logarithm domain)
    %           x: mean of mixture components
    %           P: covariance of mixture components
    
    properties
        density %density class handle
        paras   %parameters specify a PPP
    end
    
    methods
        function obj = initialize(obj,density_class_handle,birthmodel)
            %INITIATOR initializes PHDfilter class
            %INPUT: density_class_handle: density class handle
            %OUTPUT:obj.density: density class handle
            %       obj.paras.w: weights of mixture components --- vector
            %                    of size (number of mixture components x 1)
            %       obj.paras.states: parameters of mixture components ---
            %                    struct array of size (number of mixture
            %                    components x 1) 
            obj.density = density_class_handle;
            obj.paras.w = [birthmodel.w]';
            obj.paras.states = rmfield(birthmodel,'w')';
        end
        
        function obj = predict(obj,motionmodel,P_S,birthmodel)
            %PREDICT performs PPP prediction step
            %INPUT: P_S: object survival probability
            P_S_log = log(P_S);
            
            % number of birth components
            n_Hyp_b = length(birthmodel);
            
            % number of surving objects
            n_Hyp_s = length(obj.paras.w);
            
            % number of total components that shall be predicted
            n_Hyp = n_Hyp_b + n_Hyp_s;
            
            % allocate space for log weights
            w_logs = zeros(n_Hyp, 1);
            states = [];
            
            % copy the birth weights to the first weights
            w_logs(1:n_Hyp_b, 1) = arrayfun(@(b) b.w, birthmodel)';
            
            % copy the birth states
            states = arrayfun(@(b) struct('x', b.x, 'P', b.P), birthmodel)';
            
            % predict and store the components that survived
            for h=1:n_Hyp_s
                state = obj.density.predict(...
                    obj.paras.states(h), motionmodel);
                w_log = P_S_log + obj.paras.w(h);
                h_new = h + n_Hyp_b;
                w_logs(h_new,1) = w_log;
                states(h_new,1) = state;
            end
            
            obj.paras.w = w_logs;
            obj.paras.states = states;
        end
        
        function S = computeS(obj, state, measmodel)
            % compute the innovation covariance S
            %
            %Measurement model Jacobian
            Hx = measmodel.H(state.x);
            %Innovation covariance
            S = Hx * state.P * Hx' + measmodel.R;
            %Make sure matrix S is positive definite
            S = (S+S') / 2;     
        end
        
        function obj = update(obj,z,measmodel,intensity_c,P_D,gating)
            %UPDATE performs PPP update step and PPP approximation
            %INPUT: z: measurements --- matrix of size (measurement dimension 
            %          x number of measurements)
            %       intensity_c: Poisson clutter intensity --- scalar
            %       P_D: object detection probability --- scalar
            %       gating: a struct with two fields: P_G, size, used to
            %               specify the gating parameters
            
            l_0_log = log(1 - P_D);
            P_D_log = log(P_D);
            intensity_c_log = log(intensity_c);
            
            % number of measurements
            mk = size(z, 2);
            
            % number of predicted states / components
            n_states = numel(obj.paras.states);
            
            % number of update components
            % #measurements * #states + #misdetections
            n_updates = mk * n_states + n_states;
            
            % weights for all updated components
            w_logs = zeros(n_updates, 1);
            
            %% Construct update components resulted from missed detection.
            w_logs(1:n_states, 1) = l_0_log + obj.paras.w;
            % Covariance and state vector remain the same
            states = obj.paras.states; 

            %% Compute some Kalman variables            
            % since H and R are the same for all z_i the
            % predicted measurement distribution and the Kalman Gain are
            % the same values for all updates of component i. 
            z_hat = arrayfun(...
                @(state) measmodel.h(state.x), obj.paras.states, ...
                'UniformOutput', false);
            S = arrayfun(...
                @(state) obj.computeS(state, measmodel),...
                obj.paras.states, ...
                'UniformOutput', false);
            K = arrayfun(...
                @(h) obj.paras.states(h).P * measmodel.H(obj.paras.states(h).x)' / S{h}, ...
                (1:n_states)', ...
                'UniformOutput', false);
            P = arrayfun(...
                @(h) (eye(size(obj.paras.states(h).x, 1)) - K{h} * measmodel.H(obj.paras.states(h).x)) * obj.paras.states(h).P, ...
                (1:n_states)', ...
                'UniformOutput', false);
            
            i_last_state = n_states;
            
            for i=1:mk
                n_states_gated = 0;
                %w_sum = intensity_c;
                
                for h=1:n_states
                    % Perform ellipsoidal gating 
                    % for each Gaussian component in the Poisson intensity.
                    euclid_d = z(:, i) - z_hat{h};
                    mahalanobis_d = euclid_d' / S{h} * euclid_d;
                    
                    if (mahalanobis_d < gating.size)
                        n_states_gated = n_states_gated + 1;
                        i_state = i_last_state + n_states_gated;
                        x = obj.paras.states(h).x + K{h} * euclid_d;
                        updated_state = struct('x', x, 'P', P{h});
                        states(i_state, 1) = updated_state;
                        w_logs(i_state, 1) = P_D_log + obj.paras.w(h) +...
                            log_mvnpdf(z(:, i), z_hat{h}, S{h});
                        %w_sum = w_sum + exp(w_logs(i_state, 1));
                    end
                end
                % normalize
                normalize_range = i_last_state + 1:i_last_state + n_states_gated;
                %w_logs(normalize_range, 1) = ...
                %    log(exp(w_logs(normalize_range, 1)) / w_sum);
                w_log_unnormalized = zeros(n_states_gated + 1, 1);
                w_log_unnormalized(1:n_states_gated, 1) = ...
                    w_logs(normalize_range);
                w_log_unnormalized(n_states_gated + 1, 1) = intensity_c_log;
                w_logs_normalized = normalizeLogWeights(...
                    w_log_unnormalized);
                w_logs(normalize_range, 1) = w_logs_normalized(1: n_states_gated);
               
                i_last_state = i_last_state + n_states_gated;
            end
            
            obj.paras.states = states;
            obj.paras.w = w_logs(1:i_last_state, 1);
        end
        
        function obj = componentReduction(obj,reduction)
            %COMPONENTREDUCTION approximates the PPP by representing its
            %intensity with fewer parameters
            
            %Pruning
            [obj.paras.w, obj.paras.states] = hypothesisReduction.prune(obj.paras.w, obj.paras.states, reduction.w_min);
            %Merging
            if length(obj.paras.w) > 1
                [obj.paras.w, obj.paras.states] = hypothesisReduction.merge(obj.paras.w, obj.paras.states, reduction.merging_threshold, obj.density);
            end
            %Capping
            [obj.paras.w, obj.paras.states] = hypothesisReduction.cap(obj.paras.w, obj.paras.states, reduction.M);
        end
        
        function estimates = PHD_estimator(obj)
            %PHD_ESTIMATOR performs object state estimation in the GMPHD filter
            %OUTPUT:estimates: estimated object states in matrix form of
            %                  size (object state dimension) x (number of
            %                  objects) 
            % Get a mean estimate of the cardinality of objects by taking 
            % the summation of the weights of the Gaussian components  
            % (rounded to the nearest integer), denoted as n.
            n = round(sum(exp(obj.paras.w)));
            n = min(n, length(obj.paras.w));
            
            [~, best_n_idx] = sort(obj.paras.w, 'descend');
            
            estimates = cell2mat(...
                arrayfun(@(i) obj.paras.states(i, 1).x, ...
                    best_n_idx(1:n)', 'UniformOutput', false));
        end
    end
end

