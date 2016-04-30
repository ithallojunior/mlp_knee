## data must be 3xsamples and have the same size
## 3 = x, y, z
## returns angular velocity for each data item
function [av, angles, aa, vxyz] = calc_ang_v(data_point1, data_origin, data_point2, time_sample)
	av = [0];
	angles = [];
	v = 0;
	for i = [2: size(data_origin, 1)]
		## first assume data_origin as origin from others pointers
		## them translate everthing to origim
		## them calculate angular velocity
		p1 = data_point1(i-1,:) - data_origin(i-1,:);
		p2 = data_point2(i-1,:) - data_origin(i-1,:);
		angle1 = get_angle(p1, p2);
		angles = [angles; angle1];
		p1 = data_point1(i,:) - data_origin(i,:);
		p2 = data_point2(i,:) - data_origin(i,:);
		angle2 = get_angle(p1, p2);	
		v = (angle2 - angle1)/time_sample;
		av = [av; v];
	endfor
	angles = [angles; angle2];
	av(1,1) = av(2,1);

	aa = [0];
	for i = [2: size(av, 1)]
		a = (av(i) - av(i-1))/time_sample;
		aa = [aa; a];
	end

	vxyz = [ 0, 0, 0];
	for i = [2: size(data_origin, 1)]
		vxyz = [vxyz; (data_origin(i, :) - data_origin(i-1, :)) ./ time_sample];
	end
	vxyz(1, :) = vxyz(2,:);
end
