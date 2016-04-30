function nn = create_NN(ni, nh, no, number_hiddens, activation, derivative, cost_function)
	nn.ni = ni+1;
	nn.nh = nh+1;
	nn.no = no;
	nn.number_hiddens = number_hiddens;
	nn.wi = ((-0.2 - 0.2) * rand(nn.ni, nn.nh - 1)) + 0.2;
	nn.wh = ((-2 - 2) * rand(nn.nh, nn.nh-1, nn.number_hiddens -1)) + 2;
	nn.wo = ((-2 - 2) * rand(nn.nh, nn.no)) + 2;
	nn.activation = activation;
	nn.derivative = derivative;
	nn.cost_function = cost_function;
end
%!test
%!	nn = create_NN(2, 2, 2, 1, @sigmoid, @sigmoid_derivative, @sigmoid_cost_function);
%!	assert(size(nn.wh) == [3,2,0]);
%
%!test
%!	nn = create_NN(2, 2, 2, 2, @sigmoid, @sigmoid_derivative, @sigmoid_cost_function);
%!	assert(size(nn.wh) ==[3, 2]);
%!test
%!	nn = create_NN(2, 2, 2, 3, @sigmoid, @sigmoid_derivative, @sigmoid_cost_function);
%!	assert(size(nn.wh) ==[3, 2, 2]);
%!test
%!	nn = create_NN(2, 10, 2, 3, @sigmoid, @sigmoid_derivative, @sigmoid_cost_function);
%!	assert(size(nn.wh) ==[11, 10, 2]);

