function A = myLDA(Samples, Labels, NewDim)
% Input:    
%   Samples: The Data Samples 
%   Labels: The labels that correspond to the Samples
%   NewDim: The New Dimension of the Feature Vector after applying LDA

	%A=zeros(NumFeatures,NewDim);
    
	[NumSamples, NumFeatures] = size(Samples);
    NumLabels = length(Labels);
    if(NumSamples ~= NumLabels) 
        then
        fprintf('\nNumber of Samples are not the same with the Number of Labels.\n\n');
        exit
    end
    Classes = unique(Labels);
    NumClasses = length(Classes);  %The number of classes

    P = zeros(NumClasses, 1);         % Class Prior Probability
    mu = zeros(NumClasses, NumFeatures); % Class Mean
    Sw = zeros(NumFeatures, NumFeatures); % Within Class Scatter Matrix
    Sb = zeros(NumFeatures, NumFeatures); % Between Class Scatter Matrix

    %For each class i
	%Find the necessary statistics
    
    for i = 1:NumClasses
    Xi = Samples(Labels == Classes(i), :); % Samples of class i
    Ni = size(Xi, 1); % Number of samples in class i
 
    %Calculate the Class Prior Probability
	P(i)= Ni / NumSamples;
    %Calculate the Class Mean 
	mu(i,:)= mean(Xi);
    %Calculate the Within Class Scatter Matrix
    Si = cov(Xi);
	Sw= Sw + Si * P(i);%(Ni - 1);
    %Calculate the Global Mean
	m0 = mean(Samples); % Global Mean

  
    %Calculate the Between Class Scatter Matrix
    meanDiff = mu(i, :) - m0;
	Sb= Sb + P(i) * (meanDiff * meanDiff');
    end
    
    %Eigen matrix EigMat=inv(Sw)*Sb
    EigMat = inv(Sw)*Sb;
    
    %Perform Eigendecomposition
    [U, S] = svd(EigMat);

    
    %Select the NewDim eigenvectors corresponding to the top NewDim
    %eigenvalues (Assuming they are NewDim<=NumClasses-1)
	%% You need to return the following variable correctly.
	A=zeros(NumFeatures,NewDim); % Return the LDA projection vectors
    A = U(:, 1:NewDim);
     
 
end