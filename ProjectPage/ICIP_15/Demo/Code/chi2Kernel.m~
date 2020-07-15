%% Compute Chisq Distance
function [D, gamma] = func_Chi2Dist(X,Y)
    D = zeros(size(X,1),size(Y,1));
    parfor i=1:size(Y,1)
        tic;
        d = bsxfun(@minus, X, Y(i,:));
        s = bsxfun(@plus, X, Y(i,:));
        D(:,i) = sum(d.^2 ./ (s/2+eps), 2);
        toc;
        fprintf('%d th obs\n',i);
    end
    
    gamma = mean(mean(D));
    
end
