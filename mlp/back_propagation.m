function [nn, tnew] = back_propagation(nn, training, target, ao, ai, ah, deeph)

	output_delta = 0;
	ERROR = target - ao;
	output_delta = (derivative(nn, ao) .* ERROR)';

	if size(deeph, 1) == 1
		deeph_deltas = zeros(nn.nh-1, 1);
		error_deeph = nn.wo(2:end, :) * output_delta;	
		deeph_deltas = derivative(nn, deeph(1, 2:end))' .* error_deeph;
		hidden_deltas = zeros(nn.nh-1, 1);
		error_hidden = nn.wh(2:end, :) * deeph_deltas;	
		hidden_deltas = derivative(nn, ah(1, 2:end))' .* error_hidden;

		change = (output_delta * deeph)';
		nn.wo = nn.wo .+ training.N .* change + training.M .* training.co;
		con = change;

		change = (deeph_deltas * ah)';
		nn.wh = nn.wh .+ training.N .* change + training.M .* training.cdeeph;
		cdeeph = change;

		change = (hidden_deltas * ai)';
		nn.wi = nn.wi .+ training.N .* change + training.M .* training.ci;
		cin = change;

		tnew = training;
		tnew.co = con;
		tnew.cdeeph = cdeeph;
		tnew.ci = cin;
	else
		hidden_deltas = zeros(nn.nh-1, 1);
		error_hidden = nn.wo(2:end, :) * output_delta;	
		hidden_deltas = derivative(nn, ah(1, 2:end))' .* error_hidden;

		change = (output_delta * ah)';
		nn.wo = nn.wo + training.N * change + training.M * training.co;
		con = change;
		change = (hidden_deltas * ai)';
		nn.wi = nn.wi .+ training.N .* change + training.M .* training.ci;
		cin = change;
		tnew = training;
		tnew.co = con;
		tnew.ci = cin;
	end;
end

