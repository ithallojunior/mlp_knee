function [nn, J, iteration, t] = train(nn, t, pat)
	iteration = 0;
	J = [];
	m = size(pat, 1);
	init = time();
	while 1
		iteration = iteration + 1;
		e = 0;
		for item = 1:size(pat,1)
			input =  pat(item, 1:end-nn.no);
			[ao, ai, ah, deeph] = feed_forward(nn, input);
			target = pat(item, end-nn.no+1:end);
			[nn, t] = back_propagation(nn, t, target, ao, ai, ah, deeph);
			e = e + cost_function(nn, target, ao); 
		end	
		temp = e/m + t.M * (sum(sum(nn.wi(2:end, :).^2)) + sum(sum(nn.wo(2:end,:).^2)))/2*m;
		J = [J e];
		
		if t.time_to_stop == -1
			if iteration == t.iterations || (t.err ~= -1 && e <= t.err)
				break;
			end
		else
			running_time = time() - init;
			%fprintf('Running time: %f', running_time);
			if (running_time) > t.time_to_stop
				break;
			end
		end
	end
end
%!shared mlp, J, iteration
%!	mlp = create_NN(2, 2, 1, @sigmoid, @sigmoid_derivative, @sigmoid_cost_function);
%!	training_data = create_training(mlp, 0.00000000001, 1, 10000, 0.08, -1, -1);
%!	pat = xor_pattern(mlp.ni - 1);
%!	[mlp, J, iteration] = train(mlp, training_data, pat);
%!test
%!	output = feed_forward(mlp, [0 0]) > 0.5;
%!	assert(! output);
%!test
%!	assert(feed_forward(mlp, [0 1]) > 0.5);
%!test
%!	assert(feed_forward(mlp, [1 0]) > 0.5);
%!test
%!	assert(!(feed_forward(mlp, [1 1]) > 0.5));
