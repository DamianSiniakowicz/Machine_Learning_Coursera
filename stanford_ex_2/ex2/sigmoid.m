function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

if all([1 1] == size(z));
	g = 1 / (1 + exp(-z));
elseif xor(1 == size(z,1), 1 == size(z, 2));
	for i=1:length(z);
		g(i) = 1 / (1 + exp(-(z(i))));
		end;
else;
	for row_ind=1:size(z,1);
		for col_ind=1:size(z,2);
			g(row_ind,col_ind) = 1 / (1 + exp(-(z(row_ind,col_ind))));


end;
end;
end;
% =============================================================

