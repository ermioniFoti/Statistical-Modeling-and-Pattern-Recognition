function [ eigenval, eigenvec, order] = myPCA(X)
%PCA Run principal component analysis on the dataset X
%   [ eigenval, eigenvec, order] = mypca(X) computes eigenvectors of the autocorrelation matrix of X
%   Returns the eigenvectors, the eigenvalues (on diagonal) and the order 
%

% Useful values
[m, n] = size(X);

% Make sure each feature from the data is zero mean
X_centered = X - mean(X);
    
% ====================== YOUR CODE HERE ======================
Sigma = (1 / m) * (X_centered' * X_centered);
[eigenvec, D] = eig(Sigma);
eigenval = diag(D); % Vector of Eigenvalues
[eigenval, order] = sort(eigenval,'descend'); %Sorting order
eigenvec = eigenvec(:, order); %Corresponding eigenvector                     
%%eigenval = diag(eigenval);
% ========================================================================= 

end
