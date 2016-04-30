clear; clc;
nn = create_NN(8, 40, 1, 2, @tanh, @tanh_derivative, @tanh_cost_function);
t = create_training(nn, 0.0001, 0.1, 0, -1, 10 * 60 );

pat = [];
for i = 1:5
	str = strcat('dynamics_walk', mat2str(i));
	file_name = strcat(str, '.mat');
	load(str); 
	dynamics_walk = eval(str);
	data = [dynamics_walk(:,1), dynamics_walk(:, 3), dynamics_walk(:,7), dynamics_walk(:,8), dynamics_walk(:,9), dynamics_walk(:,10), dynamics_walk(:,11),  dynamics_walk(:,12), normalize(dynamics_walk(:, 2), -6, 6)];
	selected = randperm(size(data, 1));
	selected = selected(:, 1:250);
	pat = [pat; data(selected, :)];
end

[nn, J, iteration, t] = train(nn, t, pat);

save('nnt.mat', 'nn', 't', 'iteration', 'J');

figure();
plot(J);
run_gait();
