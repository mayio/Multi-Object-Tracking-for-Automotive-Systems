% My solution for the task
% Task: Write functions that implement single object trackers using
%       Nearest neighbour.
%       Probablistic data association.
%       Gaussian sum filtering.
%
classdef singleobjectracker
    %SINGLEOBJECTRACKER is a class containing functions to track a single
    %object in clutter. 
    %Model structures need to be called:
    %sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time
    %           scan, Poisson distributed --- scalar 
    %           pdf_c: clutter (Poisson) density --- scalar
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
            %INITIATOR initializes singleobjectracker class
            %INPUT: density_class_handle: density class handle
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
            %         --- scalar 
            
            obj.density = density_class_handle;
            obj.gating.P_G = P_G;
            obj.gating.size = chi2inv(obj.gating.P_G,m_d);
            obj.reduction.w_min = log(w_min);
            obj.reduction.merging_threshold = merging_threshold;
            obj.reduction.M = M;
        end
        
        function estimates = nearestNeighbourFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %NEARESTNEIGHBOURFILTER tracks a single object using nearest
            %neighbor association 
            %INPUT: state: a structure with two fields:
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of  
            %            size (measurement dimension) x (number of
            %            measurements at corresponding time step) 
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x 1
            N = numel(Z);
            estimates = cell(N, 1);
            w_theta_0 = 1 - sensormodel.P_D;
            w_theta_factor = log(sensormodel.P_D / sensormodel.intensity_c);
            
            for k=1:N
                z = Z{k};
                
                % 1. gating
                [z_ingate, ~] = obj.density.ellipsoidalGating(state, z, measmodel, obj.gating.size);
                
                if (~isempty(z_ingate))
                    % 2. calculate the predicted likelihood for each 
                    %    measurement in the gate;
                    predicted_likelihood = obj.density.predictedLikelihood(state,z_ingate,measmodel);
                    w_theta_k = predicted_likelihood + w_theta_factor;

                    % 3. compare the weight of the missed detection 
                    %    hypothesis and the weight of the object detection
                    %    hypothesis using the nearest neighbour 
                    %    measurement;
                    [max_w_theta_k, max_k] = max(w_theta_k);
                    if (w_theta_0 < max_w_theta_k)
                        state = obj.density.update(state, z_ingate(:,max_k), measmodel);
                    end
                end
                
                % 4. store the updated state
                estimates{k} = state.x;
                
                % 5. predict the new state
                state = obj.density.predict(state, motionmodel);
            end
        end
        
        
        function estimates = probDataAssocFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %PROBDATAASSOCFILTER tracks a single object using probalistic
            %data association 
            %INPUT: state: a structure with two fields:
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
            %       state dimension) x 1
            N = numel(Z);
            estimates = cell(N, 1);
            w_theta_log_0 = log(1 - sensormodel.P_D);
            w_theta_factor = log(sensormodel.P_D / sensormodel.intensity_c);
            
            for k=1:N
                z = Z{k};
                
                % 1. gating
                [z_ingate, ~] = obj.density.ellipsoidalGating(state, z, measmodel, obj.gating.size);
                
                % 2. missed detection hypothesis
                hypothesis_0 = state;
                
                % number of hypotheses without missed detection
                mk = size(z_ingate, 2);

                % Clear old hypotheses
                hypotheses_mk = repmat(state,mk + 1,1);

                % 3. calculate the predicted likelihood for each 
                %    measurement in the gate and normalize;
                log_likelihood = ...
                    obj.density.predictedLikelihood(...
                        state, z_ingate, measmodel);

                % calculate the weights
                w_theta_log_k = log_likelihood + w_theta_factor;                    

                % add the weight for the missed detection
                w_theta_log_k(mk + 1,1) = w_theta_log_0;
                w_theta_log_k = normalizeLogWeights(...
                    w_theta_log_k);

                % 4. create object detection hypotheses for each
                % detection inside the gate
                for theta = 1:mk
                    hypotheses_mk(theta,1) = obj.density.update(...
                        state, z_ingate(:,theta), measmodel);
                end

                % add the missed detection hypothesis
                hypotheses_mk(mk + 1,1) = hypothesis_0;

                % 5. prune hypotheses with small weights, and then
                % re-normalise the weights
                [w_theta_log_k, hypotheses] = ... 
                    hypothesisReduction.prune(...
                        w_theta_log_k, ...
                        hypotheses_mk, ...
                        obj.reduction.w_min);

                w_theta_log_k = normalizeLogWeights(...
                    w_theta_log_k);

                % 6. merge different hypotheses using Gaussian moment
                % matching
                [w_theta_log_k,merged_hypotheses] = ...
                    hypothesisReduction.merge(...
                        w_theta_log_k,...
                        hypotheses,...
                        10000000000000,...
                        obj.density);

                state = merged_hypotheses(1,1);
               
                % 6. store the updated state
                estimates{k} = state.x;
                
                % 7. predict the new state
                state = obj.density.predict(state, motionmodel);
            end            
        end
        
        function estimates = GaussianSumFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %GAUSSIANSUMFILTER tracks a single object using Gaussian sum
            %filtering
            %INPUT: state: a structure with two fields:
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
            %       state dimension) x 1  
            N = numel(Z);
            estimates = cell(N, 1);
            w_theta_log_0 = log(1 - sensormodel.P_D);
            w_theta_factor = log(sensormodel.P_D / sensormodel.intensity_c);
            hypotheses(1) = state;
            w_logs = [0];
            
            for k=1:N
                z = Z{k};
                n_hypotheses = numel(hypotheses);
                
                hypotheses_update = [];
                w_log_update = [];
                
                for theta = 1:n_hypotheses
                    state = hypotheses(theta);
                    w_log = w_logs(theta);
                    
                    % gating
                    [z_ingate, ~] = obj.density.ellipsoidalGating(...
                        state, z, measmodel, obj.gating.size);
                
                    % missed detection hypothesis
                    hypothesis_0 = state;
                
                    % number of hypotheses without missed detection
                    mk = size(z_ingate, 2);

                    % calculate the predicted likelihood for each 
                    % measurement in the gate;
                    log_likelihood = ...
                        obj.density.predictedLikelihood(...
                            state, z_ingate, measmodel);
                        
                    % calculate the weights
                    w_theta_log_k = w_log + log_likelihood + w_theta_factor;

                    % add the weight for the missed detection
                    w_theta_log_k(mk + 1,1) = w_log + w_theta_log_0;
                    
                    % Clear old hypotheses
                    hypotheses_mk = repmat(state,mk + 1,1);

                    % 4. create object detection hypotheses for each
                    % detection inside the gate
                    for theta = 1:mk
                        hypotheses_mk(theta,1) = obj.density.update(...
                            state, z_ingate(:,theta), measmodel);
                    end
                
                    % add the missed detection hypothesis
                    hypotheses_mk(mk + 1,1) = hypothesis_0;
                    
                    hypotheses_update = [hypotheses_update;hypotheses_mk];
                    w_log_update = [w_log_update; w_theta_log_k];
                end


                w_log_update = normalizeLogWeights(w_log_update);

                % prune hypotheses with small weights, and then
                % re-normalise the weights
                [w_log_update, hypotheses_update] = ... 
                    hypothesisReduction.prune(...
                        w_log_update, ...
                        hypotheses_update, ...
                        obj.reduction.w_min);

                w_log_update = normalizeLogWeights(...
                    w_log_update);

                % merge different hypotheses using Gaussian moment
                % matching
                [w_log_update,hypotheses_update] = ...
                    hypothesisReduction.merge(...
                        w_log_update,...
                        hypotheses_update,...
                        obj.reduction.merging_threshold,...
                        obj.density);

                % cap the number of the hypotheses, and then re-normalise
                % the weights;
                [w_logs,hypotheses] = ...
                    hypothesisReduction.cap(...
                        w_log_update,...
                        hypotheses_update,...
                        obj.reduction.M);

                w_logs = normalizeLogWeights(...
                    w_logs);
               
                % extract object state estimate using the most probably hypothesis estimation;
                [~, max_theta] = max(w_logs);
                estimates{k} = hypotheses(max_theta,1).x;
                
                % 7. predict the new state
                for theta = 1:numel(hypotheses)
                    hypotheses(theta,1) = obj.density.predict(...
                        hypotheses(theta,1), motionmodel);
                end
            end
        end
    end
end

