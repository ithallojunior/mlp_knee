function [nn, t, out, J, iteration]  = train_show(nn, t, pat)
	[nn, J, iteration] = train(nn, t, pat);
	out = [];
	for item = 1:size(pat,1)
		ao = feed_forward(nn, pat(item, 1:end-nn.no));
		out = [out; ao];
	end

	e = J(1,end)
	iteration
	figure();
	plot(J);
end
