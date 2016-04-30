function  Run_this_mlp()
clear; clc;

nn = create_NN(9, 40, 1, 2, @tanh, @tanh_derivative, @tanh_cost_function);
t = create_training(nn, 0.043142, 0.0033804, 0, -1, 5*60); %tempo
dif = 10.0;

load training_set.mat 

data = [training_set(:,1), training_set(:, 3), training_set(:,5), training_set(:,7), training_set(:,8), training_set(:,9), training_set(:,10), training_set(:,11), training_set(:,12), normalize(training_set(:, 2), -6, 6)];



selected = randperm(size(data, 1));
selected = selected(:, 1:45);
pat = data(selected, :);


[nn, J, iteration] = train(nn, t, pat);


out = [];
desired = [];
for item = 1:size(data,1)
	if not(any(item == selected))
		ao = feed_forward(nn, data(item, 1:end-nn.no));
		out = [out; ao];
		desired = [desired; data(item, end)];
	end
end

disp("\n")
SquaredMean = 0.0;
for r =1:size(out)(2)
SquaredMean =  SquaredMean +((out(r) - desired(r))^2);
endfor

SquaredMean = SquaredMean/(size(desired)(2))
J(end)
iteration

figure();
plot(J);
title('Error per iteration');
ylabel('Error');
xlabel('Iterations');
grid minor on;
figure();
plot(out, 'r');
hold on;
plot(desired, 'b');
title('Results');
legend('Obtained', 'Desired');
xlabel('Time(s)');
ylabel('Angular velocities(rad/s)');
grid minor on;
end