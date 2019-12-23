%Note that multiHypotheses does not really need to be a struct array to make the function work properly
multiHypotheses = 1:100;
%Generate some random weights
hypothesesWeight = rand(100,1);
hypothesesWeight = log(hypothesesWeight/sum(hypothesesWeight));
%Pruning
[hypothesesWeight_hat, multiHypotheses_hat] = hypothesisReduction.prune(hypothesesWeight, multiHypotheses, log(1e-2)); 
%Capping
[hypothesesWeight_hat, multiHypotheses_hat] = hypothesisReduction.cap(hypothesesWeight, multiHypotheses, 50); 