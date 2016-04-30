## Load data_load  in to a file_set friendly format. See cmack_walk.m to see format
## delete_init and delete_end, delete a number of inutilized samples from
## begin and end of the sampled data
## training_set format 4 x samples
## 4 = angular velocities and angles from left and right legs
## samples colected 
function training_set = clean_data(data_load)
	delete_init = data_load.cut_begin;
	delete_end = data_load.cut_end;
	load (strcat('original_data/', data_load.file_name))
	var = data_load.file_name(1:end-4);
	walk = eval(var);
	data = walk.Trajectories.Unidentified.Data;	
	sample_time = 1/walk.FrameRate;

	indexes = data_load.indexes;
#data(27,:, 136:207)
	[data_points, points] = get_data(data, indexes);

	data_points = data_points(:, :, [delete_init:size(data_points,3) - delete_end]);
#points

	[x, y, z] = get_data_point(data_points, find(points == indexes.tro_dir));
	points2 = [x, y, z];
	[x, y, z] = get_data_point(data_points, find(points == indexes.joe_dir));
	origin = [x, y, z];
	[x, y, z] = get_data_point(data_points, find(points == indexes.mal_lat_dir));
	points1 = [x, y, z];
#    points1
#    points2
#    origin

	[av_right, angles_right, aa_right, vxyz_right] = calc_ang_v(points1, origin, points2, sample_time);


#size(data_points)

	[x, y, z] = get_data_point(data_points, find(points == indexes.tro_esq));
	points2 = [x, y, z];

data_points(5,:,:);
	[x, y, z] = get_data_point(data_points, find(points == indexes.joe_esq));
	origin = [x, y, z];
#origin
	[x, y, z] = get_data_point(data_points, find(points == indexes.mal_lat_esq));
	points1 = [x, y, z];
	[av_left, angles_left, aa_left, vxyz_left] = calc_ang_v(points1, origin, points2, sample_time);

#points2
	training_set = [av_left, av_right, angles_left, angles_right, aa_left, aa_right, vxyz_left, vxyz_right];
end;
