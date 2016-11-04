function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

m = size(X,1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% for each observation
% calculate distance to each centroid. return index of closest centroid.

for obs_num=1:m;
	obs_i = X(obs_num,:)
	best_K = 0;
	short_dist = 0;
	for cent_num=1:K;
		cent_j = centroids(cent_num,:);
		dist_ij = sum((obs_i - cent_j).^2)
		if short_dist == 0 | dist_ij < short_dist;
			short_dist = dist_ij;
			best_K = cent_num;
		end;
	end;
	idx(obs_num) = best_K;
end;





% =============================================================

end

