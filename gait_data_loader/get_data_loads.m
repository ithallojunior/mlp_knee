function data_loads = get_data_loads()
	data_loads = [];


	## Data to training set
	data_load.file_name = 'JULIANA2701w1.mat';
# sum 1 to the value found, matlab indices starts at 1
	data_load.indexes.mal_lat_dir = 27;		#3


	data_load.indexes.joe_dir = 4;			#6
	data_load.indexes.tro_dir = 6;			#7


	data_load.indexes.mal_lat_esq = 3;		#10


	data_load.indexes.joe_esq = 16;			#13
	data_load.indexes.tro_esq = 8;			#14
	data_load.cut_begin = 136;
	data_load.cut_end = 93;
	data_loads = [data_loads; data_load];


	


end
