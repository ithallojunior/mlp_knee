## Get data from file data structures
## data is 74 x 4 x samples
## 74 is the number of channels
## 4 is x , y, z and error
## samples the number of samples from data file

function [data_points, points_loaded] = get_data(data, indexes)
	
	points_loaded = [


		indexes.mal_lat_dir,


		indexes.joe_dir,
		indexes.tro_dir,


		indexes.mal_lat_esq,


		indexes.joe_esq,
		indexes.tro_esq
	];


	data_points = data(points_loaded, 1:3, :);

end;
