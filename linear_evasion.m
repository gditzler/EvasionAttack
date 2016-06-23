function x = linear_evasion(w, x, eta, epsilon, max_iter, cap)
% x = LINEAR_EVASION(w, x, eta, epsilon, max_iter, cap)
%
% Generate a minimally evasive sample using the evasion using gradient
% based methods that were originally presented Zhang et al. (2016)
%
% INPUT 
%   w: LASSO parameter vector (linear model)
%   x: sample that we want to evade
%   eta: learning rate 
%   epsilon: small number for the stopping critera 
%   max_iter: maximum number of gradient steps
%   cap: maximum value that a feature can take. it is recommended that the
%        features are scaled to similar domains.
%
% OUTPUT 
%   x: new version of x that evades the original 
% 
% REFERENCE 
%   Zhang et al., "Adversarial Feature Selection Against Evasion Attacks,"
%     IEEE Transactions on Cybernetics, 2016. 
%
% By: Gregory Ditzler (gregory.ditzler@gmail.com)
%
x_orig = x;
for m = 1:max_iter
  x_old = x;
  if x*w > 0 
    % move closer to the decision boundary 
    x = x_old - eta*w';
  else
    % decrease the cost function 
    x = x_old - 2*eta*(x_old-x_orig);
  end
  % stay within the domain
  x(x>cap) = cap;
  x(x<-cap) = -cap;
  x(end) = 1;
  
  if norm(x-x_orig) - norm(x_old-x_orig) < epsilon
    if m ~= 1
      % do at least 2 steps before stopping. 
      break;
    end
  end
end
