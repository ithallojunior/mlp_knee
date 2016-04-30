function J = cost_function(nn, target, output)
	J = 0;
	for index = 1:size(output, 2)
		J = J + nn.cost_function(index, target, output); 
	end
end
