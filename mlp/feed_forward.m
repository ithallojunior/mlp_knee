function [output, ai, ah, deeph] = feed_forward(nn, input)
	ai = [1, input]; %bias added
	ah = [1, activation(nn, ai, nn.wi)]; %bias added
	deeph = [];
	if nn.number_hiddens > 1
		temp = ah;
		for index = [2:nn.number_hiddens]
			if nn.number_hiddens == 2
				w = nn.wh;
			else
				w = nn.wh(:,:,index - 1);
			end
			temp = [1, activation(nn, temp, w)];
			deeph = [deeph; temp];

		end	
		ao = activation(nn, temp, nn.wo);
	else
		ao = activation(nn, ah, nn.wo);
	end
	output = ao;
end
%!test 
%!	nn = create_NN(2, 2, 2, 1, @sigmoid, @sigmoid_derivative, @sigmoid_cost_function);
%!	[output, ai, ah, deeph] = feed_forward(nn,[0 0]);
%!	assert(size(deeph), [0 0]);
%!test
%!	nn = create_NN(2, 2, 2, 2, @sigmoid, @sigmoid_derivative, @sigmoid_cost_function);
%!	[output, ai, ah, deeph] = feed_forward(nn,[0 0]);
%!	assert(size(deeph), [1 3]);
%!test
%!	nn = create_NN(2, 2, 2, 3, @sigmoid, @sigmoid_derivative, @sigmoid_cost_function);
%!	[output, ai, ah, deeph] = feed_forward(nn,[0 0]);
%!	assert(size(deeph), [2, 3]);
%!test
%!	nn = create_NN(2, 2, 2, 4, @sigmoid, @sigmoid_derivative, @sigmoid_cost_function);
%!	[output, ai, ah, deeph] = feed_forward(nn,[0 0]);
%!	assert(size(deeph), [3, 3]);
