function run_gait()
	clear all; close all; clc;
	load nnt.mat
	plot(J);
	fprintf('iteration: %d', iteration);
	for i = 1:5
		str = strcat('dynamics_walk', mat2str(i));
		file_name = strcat(str, '.mat');
		load(str); 
		dynamics_walk = eval(str);
		data = [dynamics_walk(:,1), dynamics_walk(:, 3), dynamics_walk(:,7), dynamics_walk(:,8), dynamics_walk(:,9), dynamics_walk(:,10), dynamics_walk(:,11),  dynamics_walk(:,12), normalize(dynamics_walk(:, 2), -6, 6)];

		out = [];
		for item = 1:size(data,1)
			ao = feed_forward(nn, data(item, 1:end-1));
			out = [out; ao];
		end

		desired = data(:,end);
		figure();
		plot(out, 'r');
		hold on;
		plot(desired, 'b');
		legend('nn', 'desired');
	end
end
