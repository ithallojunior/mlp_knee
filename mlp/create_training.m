function t = create_training(nn, N, M, iterations, err, time_to_stop)
	t.N = N;
	t.M = M;
	t.iterations = iterations;
	t.ci = zeros(nn.ni, nn.nh -1);
	t.co = zeros(nn.nh, nn.no);
	t.cdeeph = zeros(nn.nh, nn.nh-1, nn.number_hiddens-1);
	t.err = err;
	t.time_to_stop = time_to_stop;
end
