classdef PMBMfilter
    %PMBMFILTER is a class containing necessary functions to implement the
    %PMBM filter
    %Model structures need to be called:
    %    sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time scan, 
    %                     Poisson distributed --- scalar
    %           pdf_c: value of clutter pdf --- scalar
    %           intensity_c: Poisson clutter intensity --- scalar
    %       motionmodel: a structure specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %       measmodel: a structure specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the target state
    %           R: measurement noise covariance matrix
    %       birthmodel: a structure array specifies the birth model (Gaussian
    %       mixture density) parameters --- (1 x number of birth components)
    %           w: weights of mixture components (in logarithm domain)
    %           x: mean of mixture components
    %           P: covariance of mixture components
    properties
        density %density class handle
        paras   %%parameters specify a PMBM
    end
    
    methods
        function obj = initialize(obj,density_class_handle,birthmodel)
            %INITIATOR initializes PMBMfilter class
            %INPUT: density_class_handle: density class handle
            %       birthmodel: a struct specifying the intensity (mixture)
            %       of a PPP birth model
            %OUTPUT:obj.density: density class handle
            %       obj.paras.PPP.w: weights of mixture components in PPP
            %       intensity --- vector of size (number of mixture
            %       components x 1) in logarithmic scale
            %       obj.paras.PPP.states: parameters of mixture components
            %       in PPP intensity struct array of size (number of
            %       mixture components x 1)
            %       obj.paras.MBM.w: weights of MBs --- vector of size
            %       (number of MBs (global hypotheses) x 1) in logarithmic 
            %       scale
            %       obj.paras.MBM.ht: hypothesis table --- matrix of size
            %       (number of global hypotheses x number of hypothesis
            %       trees). Entry (h,i) indicates that the (h,i)th local
            %       hypothesis in the ith hypothesis tree is included in
            %       the hth global hypothesis. If entry (h,i) is zero, then
            %       no local hypothesis from the ith hypothesis tree is
            %       included in the hth global hypothesis.
            %       obj.paras.MBM.tt: local hypotheses --- cell of size
            %       (number of hypothesis trees x 1). The ith cell contains
            %       local hypotheses in struct form of size (number of
            %       local hypotheses in the ith hypothesis tree x 1). Each
            %       struct has two fields: r: probability of existence;
            %       state: parameters specifying the object density
            
            obj.density = density_class_handle;
            obj.paras.PPP.w = [birthmodel.w]';
            obj.paras.PPP.states = rmfield(birthmodel,'w')';
            obj.paras.MBM.w = [];
            obj.paras.MBM.ht = [];
            obj.paras.MBM.tt = {};
        end
        
        function Bern = Bern_predict(obj,Bern,motionmodel,P_S)
            %BERN_PREDICT performs prediction step for a Bernoulli component
            %INPUT: Bern: a struct that specifies a Bernoulli component,
            %             with fields: r: probability of existence ---
            %                          scalar;
            %                          state: a struct contains parameters
            %                          describing the object pdf
            %       P_S: object survival probability
            
            % Probability of survival * Probability of existence
            Bern.r = P_S * Bern.r;

            % Kalman prediction of new state
            Bern.state = obj.density.predict(Bern.state, motionmodel);
        end
        
        function [Bern, lik_undetected] = Bern_undetected_update(obj,tt_entry,P_D)
            %BERN_UNDETECTED_UPDATE calculates the likelihood of missed
            %detection, and creates new local hypotheses due to missed
            %detection.
            %INPUT: tt_entry: a (2 x 1) array that specifies the index of
            %       local hypotheses. (i,j) indicates the jth local
            %       hypothesis in the ith hypothesis tree. 
            %       P_D: object detection probability --- scalar
            %OUTPUT:Bern: a struct that specifies a Bernoulli component,
            %       with fields: r: probability of existence --- scalar;
            %                    state: a struct contains parameters
            %                    describing the object pdf
            %       lik_undetected: missed detection likelihood --- scalar
            %       in logorithmic scale
            
            % misdetection likelihood
            l0 = 1 - P_D;

            % i-th hypothesis tree, j-th local hypothesis
            i = tt_entry(1);
            j = tt_entry(2);

            % local hypotheses
            lhs = obj.paras.MBM.tt{i, 1};

            % local hypothesis
            lh = lhs(j);

            % the state remains the same
            Bern = lh;

            % predicted likelihood
            lik_undetected = (1 - lh.r + lh.r * l0);

            % update probability of existence
            Bern.r = (lh.r * l0) / lik_undetected;

            % predicted log likelihood
            lik_undetected = log(lik_undetected);
        end
        
        function lik_detected = Bern_detected_update_lik(obj,tt_entry,z,measmodel,P_D)
            %BERN_DETECTED_UPDATE_LIK calculates the predicted likelihood
            %for a given local hypothesis. 
            %INPUT: tt_entry: a (2 x 1) array that specifies the index of
            %       local hypotheses. (i,j) indicates the jth
            %       local hypothesis in the ith hypothesis tree.
            %       z: measurement array --- (measurement dimension x
            %       number of measurements)
            %       P_D: object detection probability --- scalar
            %OUTPUT:lik_detected: predicted likelihood --- (number of
            %measurements x 1) array in logarithmic scale 
            
            % i-th hypothesis tree, j-th local hypothesis
            i = tt_entry(1);
            j = tt_entry(2);

            % local hypotheses
            lhs = obj.paras.MBM.tt{i, 1};

            % local hypothesis
            lh = lhs(j);

            % predicted log-likelihood
            lik_detected = obj.density.predictedLikelihood(...
                lh.state, z, measmodel) + log(P_D) + log(lh.r);
        end
        
        function Bern = Bern_detected_update_state(obj,tt_entry,z,measmodel)
            %BERN_DETECTED_UPDATE_STATE creates the new local hypothesis
            %due to measurement update. 
            %INPUT: tt_entry: a (2 x 1) array that specifies the index of
            %                 local hypotheses. (i,j) indicates the jth
            %                 local hypothesis in the ith hypothesis tree.
            %       z: measurement vector --- (measurement dimension x 1)
            %OUTPUT:Bern: a struct that specifies a Bernoulli component,
            %             with fields: r: probability of existence ---
            %                          scalar; 
            %                          state: a struct contains parameters
            %                          describing the object pdf 


            % i-th hypothesis tree, j-th local hypothesis
            i = tt_entry(1);
            j = tt_entry(2);

            % local hypotheses
            lhs = obj.paras.MBM.tt{i, 1};

            % local hypothesis
            lh = lhs(j);

            % copy the state to use it as template
            Bern = lh;
            
            % state update of the Bernoulli
            Bern.state = obj.density.update(lh.state, z, measmodel);
            Bern.r = 1;
        end
        
        function obj = PPP_predict(obj,motionmodel,birthmodel,P_S)
            %PPP_PREDICT performs predicion step for PPP components
            %hypothesising undetected objects.
            %INPUT: P_S: object survival probability --- scalar

            % Predict the undetected objects
            predict_undetected = @(i) ...
                obj.density.predict(...
                    obj.paras.PPP.states(i,1), motionmodel);

            obj.paras.PPP.states = arrayfun(...
                predict_undetected, [1:length(obj.paras.PPP.states)]');
            obj.paras.PPP.w = obj.paras.PPP.w + log(P_S);

            % Add birth components
            obj.paras.PPP.states = ...
                [obj.paras.PPP.states;rmfield(birthmodel, 'w')'];
            obj.paras.PPP.w = ...
                [obj.paras.PPP.w; [birthmodel.w]'];
        end
        
        function [Bern, lik_new] = PPP_detected_update(obj,indices,z,measmodel,P_D,clutter_intensity)
            %PPP_DETECTED_UPDATE creates a new local hypothesis by
            %updating the PPP with measurement and calculates the
            %corresponding likelihood.
            %INPUT: z: measurement vector --- (measurement dimension x 1)
            %       P_D: object detection probability --- scalar
            %       clutter_intensity: Poisson clutter intensity --- scalar
            %       indices: boolean vector, if measurement z is inside the
            %       gate of mixture component i, then indices(i) = true
            %OUTPUT:Bern: a struct that specifies a Bernoulli component,
            %             with fields: r: probability of existence ---
            %             scalar;
            %             state: a struct contains parameters describing
            %             the object pdf
            %       lik_new: predicted likelihood of PPP --- scalar in
            %       logarithmic scale
            
            % For each mixture component in the PPP intensity, perform 
            % Kalman update and calculate the predicted likelihood for 
            % each detection inside the corresponding ellipsoidal gate.
            clutter_intensity_log = log(clutter_intensity);
            P_D_log = log(P_D);
            
            ppp_states_gated = obj.paras.PPP.states(indices,1);
            ppp_states_updated = arrayfun(...
                @(state) obj.density.update(state, z, measmodel), ...
                ppp_states_gated);
            w_log_gated = obj.paras.PPP.w(indices,1);
            
            % compute predicted likelihood
            
            n_gated = length(ppp_states_gated);
            w_tilde_logs = zeros(n_gated, 1);
            for (i = 1:n_gated)
                state = ppp_states_gated(i,1);
                w_log = w_log_gated(i);
                Hx = measmodel.H(state.x);
                mu = measmodel.h(state.x);
                S = Hx * state.P * Hx' + measmodel.R;
                %Make sure matrix S is positive definite
                S = (S+S')/2;
                w_tilde_logs(i, 1) = w_log + P_D_log + log_mvnpdf(...
                    z, ...
                    mu, ...
                    S);
            end
                
            %l = arrayfun(...
            %    @(state) log_mvnpdf(...
            %        z, ...
            %        measmodel.h(state.x), ...
            %        measmodel.H(state.x) * state.P * measmodel.H(state.x)' + measmodel.R), ...
            %    ppp_states_gated);
           
            [w_logs, rho] = normalizeLogWeights(w_tilde_logs);

            % The returned likelihood should be the sum of the predicted 
            % likelihoods calculated for each mixture component in the 
            % PPP intensity and the clutter intensity. (You can make use 
            % of the normalizeLogWeights function to achieve this.)
            [~, lik_new] = normalizeLogWeights([w_tilde_logs; clutter_intensity_log]);
            
            % Perform Gaussian moment matching for the updated object 
            % state densities resulted from being updated by the 
            % same detection.
            merged_state = obj.density.momentMatching(w_logs, ppp_states_updated);

            % The returned existence probability of the Bernoulli 
            % component is the ratio between the sum of the predicted 
            % likelihoods and the returned likelihood. (Be careful that 
            % the returned existence probability is in decimal scale while
            % the likelihoods you calculated beforehand are in logarithmic
            % scale.)
            r = exp(rho - lik_new);
            Bern.r = r;
            Bern.state = merged_state;
        end
        
        function obj = PPP_undetected_update(obj,P_D)
            %PPP_UNDETECTED_UPDATE performs PPP update for missed detection.
            %INPUT: P_D: object detection probability --- scalar
            obj.paras.PPP.w = ...
                obj.paras.PPP.w + log(1 - P_D);
            
        end
        
        function obj = PPP_reduction(obj,prune_threshold,merging_threshold)
            %PPP_REDUCTION truncates mixture components in the PPP
            %intensity by pruning and merging
            %INPUT: prune_threshold: pruning threshold --- scalar in
            %       logarithmic scale
            %       merging_threshold: merging threshold --- scalar
            [obj.paras.PPP.w, obj.paras.PPP.states] = hypothesisReduction.prune(obj.paras.PPP.w, obj.paras.PPP.states, prune_threshold);
            if ~isempty(obj.paras.PPP.w)
                [obj.paras.PPP.w, obj.paras.PPP.states] = hypothesisReduction.merge(obj.paras.PPP.w, obj.paras.PPP.states, merging_threshold, obj.density);
            end
        end
        
        function obj = Bern_recycle(obj,prune_threshold,recycle_threshold)
            %BERN_RECYCLE recycles Bernoulli components with small
            %probability of existence, adds them to the PPP component, and
            %re-index the hypothesis table. If a hypothesis tree contains no
            %local hypothesis after pruning, this tree is removed. After
            %recycling, merge similar Gaussian components in the PPP
            %intensity
            %INPUT: prune_threshold: Bernoulli components with probability
            %       of existence smaller than this threshold are pruned ---
            %       scalar
            %       recycle_threshold: Bernoulli components with probability
            %       of existence smaller than this threshold needed to be
            %       recycled --- scalar
            
            n_tt = length(obj.paras.MBM.tt);
            for i = 1:n_tt
                idx = arrayfun(@(x) x.r<recycle_threshold & x.r>=prune_threshold, obj.paras.MBM.tt{i});
                if any(idx)
                    %Here, we should also consider the weights of different MBs
                    idx_t = find(idx);
                    n_h = length(idx_t);
                    w_h = zeros(n_h,1);
                    for j = 1:n_h
                        idx_h = obj.paras.MBM.ht(:,i) == idx_t(j);
                        [~,w_h(j)] = normalizeLogWeights(obj.paras.MBM.w(idx_h));
                    end
                    %Recycle
                    temp = obj.paras.MBM.tt{i}(idx);
                    obj.paras.PPP.w = [obj.paras.PPP.w;log([temp.r]')+w_h];
                    obj.paras.PPP.states = [obj.paras.PPP.states;[temp.state]'];
                end
                idx = arrayfun(@(x) x.r<recycle_threshold, obj.paras.MBM.tt{i});
                if any(idx)
                    %Remove Bernoullis
                    obj.paras.MBM.tt{i} = obj.paras.MBM.tt{i}(~idx);
                    %Update hypothesis table, if a Bernoulli component is
                    %pruned, set its corresponding entry to zero
                    idx = find(idx);
                    for j = 1:length(idx)
                        temp = obj.paras.MBM.ht(:,i);
                        temp(temp==idx(j)) = 0;
                        obj.paras.MBM.ht(:,i) = temp;
                    end
                end
            end
            
            %Remove tracks that contains no valid local hypotheses
            idx = sum(obj.paras.MBM.ht,1)~=0;
            obj.paras.MBM.ht = obj.paras.MBM.ht(:,idx);
            obj.paras.MBM.tt = obj.paras.MBM.tt(idx);
            if isempty(obj.paras.MBM.ht)
                %Ensure the algorithm still works when all Bernoullis are
                %recycled
                obj.paras.MBM.w = [];
            end
            
            %Re-index hypothesis table
            n_tt = length(obj.paras.MBM.tt);
            for i = 1:n_tt
                idx = obj.paras.MBM.ht(:,i) > 0;
                [~,~,obj.paras.MBM.ht(idx,i)] = unique(obj.paras.MBM.ht(idx,i),'rows','stable');
            end
            
            %Merge duplicate hypothesis table rows
            if ~isempty(obj.paras.MBM.ht)
                [ht,~,ic] = unique(obj.paras.MBM.ht,'rows','stable');
                if(size(ht,1)~=size(obj.paras.MBM.ht,1))
                    %There are duplicate entries
                    w = zeros(size(ht,1),1);
                    for i = 1:size(ht,1)
                        indices_dupli = (ic==i);
                        [~,w(i)] = normalizeLogWeights(obj.paras.MBM.w(indices_dupli));
                    end
                    obj.paras.MBM.ht = ht;
                    obj.paras.MBM.w = w;
                end
            end
            
        end
        
        function obj = PMBM_predict(obj,P_S,motionmodel,birthmodel)
            %PMBM_PREDICT performs PMBM prediction step.
            %Bern_predict
            nHypTrees = length(obj.paras.MBM.tt);
            
            for iHypTree = 1:nHypTrees
                obj.paras.MBM.tt{iHypTree,1} = arrayfun(...
                    @(hyp) obj.Bern_predict(hyp, motionmodel,P_S), ...
                    obj.paras.MBM.tt{iHypTree,1});
            end
            
            obj = obj.PPP_predict(motionmodel,birthmodel,P_S);
        end
        
        function obj = PMBM_update(obj,z,measmodel,sensormodel,gating,w_min,M)
            %PMBM_UPDATE performs PMBM update step.
            %INPUT: z: measurements --- array of size (measurement
            %       dimension x number of measurements)
            %       gating: a struct with two fields that specifies gating
            %       parameters: P_G: gating size in decimal --- scalar;
            %                   size: gating size --- scalar.
            %       wmin: hypothesis weight pruning threshold --- scalar in
            %       logarithmic scale
            %       M: maximum global hypotheses kept
            
            % 1. Perform ellipsoidal gating for each Bernoulli state 
            %    density and each mixture component in the PPP intensity.
            % 2. Bernoulli update. For each Bernoulli state density, create
            %    a misdetection hypothesis (Bernoulli component), and 
            %    m object detection hypothesis (Bernoulli component), 
            %    where m is the number of detections inside the ellipsoidal
            %    gate of the given state density.
            % 3. Update PPP with detections. Note that for detections that
            %    are not inside the gate of undetected objects, create 
            %    dummy Bernoulli components with existence probability 
            %    r = 0; in this case, the corresponding likelihood is 
            %    simply the clutter intensity.
            % 4. For each global hypothesis, construct the corresponding 
            %    cost matrix and use Murty's algorithm to obtain the M best
            %    global hypothesis with highest weights. Note that for 
            %    detections that are only inside the gate of undetected 
            %    objects, they do not need to be taken into account when
            %    forming the cost matrix.
            % 5. Update PPP intensity with misdetection.
            % 6. Update the global hypothesis look-up table.
            % 7. Prune global hypotheses with small weights and 
            %    cap the number.
            % 8. Prune local hypotheses (or hypothesis trees) that do not
            %    appear in the maintained global hypotheses, and re-index
            %    the global hypothesis look-up table.

            % measurement count
            mk = size(z, 2);
            
            % count of global hypothesis
            n_tt = length(obj.paras.MBM.tt);
            
            % Perform ellipsoidal gating for each 
            % mixture component in the PPP intensity.
            [~, ppp_ingate_cells] = arrayfun(...
                @(state) obj.density.ellipsoidalGating(...
                    state, z, measmodel, gating.size),...
                obj.paras.PPP.states, ...
                'UniformOutput',false);
            
            ppp_ingate = cell2mat(ppp_ingate_cells');
            bern_ingate_cells = {};

            % Perform ellipsoidal gating for each Bernoulli state 
            % density
            
            % for each hypothesis tree
            z_to_bernoulli = false(mk,1);
            for i = 1:n_tt
                % local hypothesis
                lhs = obj.paras.MBM.tt{i, 1};
                n_lhs = length(lhs);
                
                % for each leaf (local hypothesis) do gating
                bern_ingate_cells{i} = zeros(mk,n_lhs);

                % each gating cell corresponds to a local hypothesis tree
                % where each cell contains a matrix of mk rows and n_lhs
                % columns
                for i_lh = 1:n_lhs
                    % gating
                    [~, bern_ingate_cells{i}(:,i_lh)] = ... 
                            obj.density.ellipsoidalGating(...
                                lhs(i_lh).state, z, measmodel, gating.size);
                    z_to_bernoulli = ... 
                        z_to_bernoulli | bern_ingate_cells{i}(:,i_lh);
                end
            end
            
            
            is_ppp_ingate = sum(ppp_ingate, 2) > 0;            
            ppp_only_ingate =  is_ppp_ingate > z_to_bernoulli;            
            is_ppp_or_bern_ingate = is_ppp_ingate | z_to_bernoulli;
            
            ppp_and_bern_ingate = ppp_ingate(z_to_bernoulli, :);
            is_ppp_and_bern_ingate = sum(ppp_and_bern_ingate, 2) > 0;
            
            z_bern = z(:,z_to_bernoulli);
            mk = size(z_bern, 2);
            
            bern_ingate_cells = cellfun(...
                @(z_masks_i) z_masks_i(z_to_bernoulli,:), ...
                bern_ingate_cells, 'UniformOutput' ,false);
            
            % Bernoulli update. For each Bernoulli state density,
            % create a misdetection hypothesis (Bernoulli component),
            % and m object detection hypothesis (Bernoulli component), 
            % where m is the number of detections inside the
            % ellipsoidal gate of the given state density.

            % number of hypothesis trees (number of available trees plus
            % number of trees spawned by measurements
            % 
            n_tt_up = n_tt + mk; 
            hyp_trees = cell(n_tt_up, 1);
            tt_lik = cell(n_tt, 1);
            
            % for each hypothesis tree
            for i = 1:n_tt
                % hypothesis tree i
                lhs = obj.paras.MBM.tt{i, 1};
                
                % number of leafs / number of local hypotheses
                n_lhs = length(lhs);
                
                % initialize log likelihoods with infinity
                tt_lik{i} = -inf(n_lhs, mk + 1);
                hyp_trees{i} = cell(n_lhs * (mk + 1), 1);
                
                for (j = 1:n_lhs)
                    lh = lhs(j);
                    tt_entry = [i, j];
                    z_ingate_j = bern_ingate_cells{i}(:, j) > 0;
                    
                    % misdetection
                    [bern_undetected, lik_undetected] = ... 
                        obj.Bern_undetected_update(...
                            tt_entry, sensormodel.P_D);
                    
                    hyp_trees{i}{(j - 1) * (mk + 1) + 1} = bern_undetected;
                    tt_lik{i}(j, 1) = lik_undetected;
                    
                    %
                    % detection of bernouli with measurement
                    %

                    % predicted likelihood --- (number of
                    % measurements x 1) array in logarithmic scale
                    lik_detected = obj.Bern_detected_update_lik(...
                        tt_entry, z_bern(:, z_ingate_j), measmodel, sensormodel.P_D);
                    
                    tt_lik{i}(j, [false;z_ingate_j]) = lik_detected;
                    
                    for z_i = 1:mk
                        if z_ingate_j(z_i, 1)
                            hyp_trees{i}{(j - 1) * (mk + 1) + z_i + 1} = ...
                                obj.Bern_detected_update_state(...
                                    tt_entry, z_bern(:,z_i), measmodel);
                        end
                    end
                end
            end
            

            % Update PPP with detections. Note that for detections that
            % are not inside the gate of undetected objects, create 
            % dummy Bernoulli components with existence probability 
            % r = 0; in this case, the corresponding likelihood is 
            % simply the clutter intensity.  
            %
            % Each measurement creates a new hypothesis tree

            Bern_undetected_lik = -inf(mk, 1);
            
            for (z_i = 1:mk)
                z_i_ingate = ppp_and_bern_ingate(z_i, :);
                
                if (is_ppp_and_bern_ingate(z_i, 1))
                    [Bern_PPP_detected_update, lik_new_PPP_detected_update] = ...
                        obj.PPP_detected_update(...
                            z_i_ingate, z_bern(:, z_i), measmodel, ...
                            sensormodel.P_D, sensormodel.intensity_c);
                    hyp_trees{n_tt + z_i, 1}{1} = Bern_PPP_detected_update;
                    Bern_undetected_lik(z_i, 1) = ...
                        lik_new_PPP_detected_update;
                else
                    % there are already dummy bernoullis, no need to 
                    % create some
                    Bern_undetected_lik(z_i) = ...
                        log(sensormodel.intensity_c);
                end
            end
           
            % For each global hypothesis, construct the corresponding 
            % cost matrix and use Murty's algorithm to obtain the M best
            % global hypothesis with highest weights. Note that for 
            % detections that are only inside the gate of undetected 
            % objects, they do not need to be taken into account when
            % forming the cost matrix.
            
            n_H = length(obj.paras.MBM.w);
            new_global_H = zeros(0, n_tt_up);
            new_w_log_h = [];
            n_H_up = 0;
            
            L_u = inf(mk);
            
            % association between measurements and previously 
            % undetected objects (clutter or new objects)
            for i = 1:mk
                L_u(i, i) = -Bern_undetected_lik(i, 1);
            end
            
            if (n_H == 0)
                new_w_log_h = 0;
                n_H_up = 1;
                new_global_H = zeros(1, mk);
                new_global_H(1, is_ppp_and_bern_ingate) = 1;
            end
            
            for H_i = 1:n_H
                % Create 2D cost matrix; 
                % 
                % number of measurements rows, 
                % number of bernoullis (association between measurements
                % and previously detected objects) + number of
                % measurements columns (associations between measurements
                % and previously undetected objects - clutter or new
                % objects)
                %
                
                L_d = inf(mk, n_tt);
                sum_l0 = 0;

                % association between measurements and previously 
                % detected objects
                for i = 1:n_tt
                    local_h_i = obj.paras.MBM.ht(H_i, i);
                    if (local_h_i > 0)
                        lik_undetected =  tt_lik{i}(local_h_i,1);
                        L_d(1:mk, i) = ... 
                            -(tt_lik{i}(local_h_i, 2:end) - lik_undetected);
                        sum_l0 = sum_l0 + lik_undetected;
                    end
                end
                
                L = [L_d L_u];
                
                if isempty(L)
                    % in case there are no associations
                    gainBest = 0;
                    col4rowBest = 0;
                else
                    % obtain M best assignments using a provided  
                    % M-best 2D assignment solver; 

                    % col4rowBest: A numRowXk vector where the entry in each 
                    %    element is an assignment of the element in that row 
                    %    to a column. 0 entries signify unassigned rows.
                    %    The columns of this matrix are the hypotheses.
                    [col4rowBest,~,gainBest] = kBest2DAssign(L, M);
                    assert(all(gainBest ~= -1), ...
                        'Assignment problem is not possible to solve');
                end

                % update weights
                new_w_log_h = [new_w_log_h; ...
                    -gainBest + sum_l0 + obj.paras.MBM.w(H_i)];

                % there might be not as many hypotheses available
                M_left = length(gainBest);
                
                % update global hypothesis look-up table according 
                % to the M best assignment matrices obtained and 
                % use your new local hypotheses indexing;
                new_global_H_h = zeros(M_left, n_tt_up);
                
                for M_i=1:M_left
                    new_global_H_h(M_i, :) = 0;
                    % we use the prior look-up-table, the prior
                    % hypothesis and the data association to figure out
                    % which posterior local hypothesis is included in
                    % the look-up-table
                    for i = 1:n_tt
                        if obj.paras.MBM.ht(H_i, i) > 0
                            tree_node_idx = ...
                                find(col4rowBest(:, M_i) == i, 1);
                            
                            if isempty(tree_node_idx)
                                % missed detection hypothesis
                                new_global_H_h(M_i, i) = ...
                                    (obj.paras.MBM.ht(H_i, i) - 1) * (mk + 1) + 1;
                            else
                                % measurement update hypothesis
                                new_global_H_h(M_i,i) = ...
                                    (obj.paras.MBM.ht(H_i, i) - 1) * (mk + 1) + tree_node_idx + 1;
                            end
                        end
                    end
                    
                    for i = n_tt + 1:n_tt_up
                        idx = find(col4rowBest(:, M_i) == i, 1);
                        
                        if ~isempty(idx) && is_ppp_and_bern_ingate(idx)
                            % measurement update for PPP
                            new_global_H_h(M_i, i) = 1;
                        end
                    end
                end

                n_H_up = n_H_up + M_left;
                new_global_H = [new_global_H; new_global_H_h];
            end
            
            % The hypothesis created by measurements that were not
            % associated to any detected object but to undetected objects
            % must be added to the hypothesis tree
            
            z_to_ppp = z(:, ppp_only_ingate);
            n_z_to_ppp = size(z_to_ppp, 2);
            ppp_only_ingate_matrix = ppp_ingate(ppp_only_ingate, :);
            
            for i = 1:n_z_to_ppp
                [hyp_trees{n_tt_up + i, 1}{1}, ~] = ...
                    obj.PPP_detected_update(...
                        ppp_only_ingate_matrix(i, :),...
                        z_to_ppp(:, i),...
                        measmodel,...
                        sensormodel.P_D,...
                        sensormodel.intensity_c);
            end
            new_global_H = [new_global_H ones(n_H_up, n_z_to_ppp)];
            
            obj = obj.PPP_undetected_update(sensormodel.P_D);

            % Prune global hypotheses with small weights and 
            % cap the number.
            %
            % Prune local hypotheses (or hypothesis trees) that do not
            % appear in the maintained global hypotheses, and re-index
            % the global hypothesis look-up table.
            [new_w_log_h, hyp_left] = hypothesisReduction.prune(...
                new_w_log_h, 1:n_H_up, w_min );
            new_global_H = new_global_H(hyp_left, :);

            % capping
            [new_w_log_h, hyp_left] = hypothesisReduction.cap(...
                new_w_log_h, 1:length(new_w_log_h), M);
            new_global_H = new_global_H(hyp_left, :);

            %normalize log weights
            new_w_log_h = normalizeLogWeights(new_w_log_h);
            
            % prune all global hypothesis that don't 
            % reference trees
            if ~isempty(new_global_H)
                idx = sum(new_global_H, 1) > 0;
                new_global_H = new_global_H(:,idx);
                hyp_trees = hyp_trees(idx);
                n_tt_up = size(new_global_H, 2);
            end            
            
            % clear old tree
            obj.paras.MBM.tt = cell(n_tt_up, 1);

            for i = 1:n_tt_up            
                % prune local hypotheses that are not included in any
                % of the global hypotheses;
                hyp = new_global_H(:, i);
                hyp_keep = unique(hyp(hyp ~= 0), 'stable');
                pruned_hyp_tree = hyp_trees{i}(hyp_keep);
                obj.paras.MBM.tt{i} = [pruned_hyp_tree{:}]';
            end
                
            % re-index the global hypothesis table
            for i = 1:n_tt_up
                idx = new_global_H(:,i) > 0;
                [~,~,new_global_H(idx, i)] = ...
                    unique(new_global_H(idx, i), 'rows', 'stable');
            end
            
            obj.paras.MBM.w = new_w_log_h;
            obj.paras.MBM.ht = new_global_H;
        end
        
        function estimates = PMBM_estimator(obj,threshold)
            %PMBM_ESTIMATOR performs object state estimation in the PMBM
            %filter
            %INPUT: threshold (if exist): object states are extracted from
            %       Bernoulli components with probability of existence no
            %       less than this threshold in Estimator 1. Given the
            %       probabilities of detection and survival, this threshold
            %       determines the number of consecutive misdetections
            %OUTPUT:estimates: estimated object states in matrix form of
            %       size (object state dimension) x (number of objects)
            %%%
            %First, select the multi-Bernoulli with the highest weight.
            %Second, report the mean of the Bernoulli components whose
            %existence probability is above a threshold. 
            estimates = [];
            [~, best_mb_idx] = max(obj.paras.MBM.w);
            best_h = obj.paras.MBM.ht(best_mb_idx,:);
            n_B = length(best_h);
            i_est = 0;
            
            for i_B = 1:n_B
                b_idx = best_h(i_B);
                
                if (b_idx > 0)
                    % for each state the best bernoulli
                    bern = obj.paras.MBM.tt{i_B}(b_idx);
                    
                    if bern.r > threshold
                        i_est = i_est + 1;
                        estimates(:, i_est) = bern.state.x;
                    end
                end
            end
        end
    end
end