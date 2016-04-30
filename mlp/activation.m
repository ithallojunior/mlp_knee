%	activations: Vector(1xi) i = number inputs pluss bias;
%	weights: Matrix(ixj) j = number activations of next layer minus bias. 
function A = activation(nn, inputs, weights)
	%A = tanh(activations * weights);
	A = nn.activation(inputs*weights);
end 
