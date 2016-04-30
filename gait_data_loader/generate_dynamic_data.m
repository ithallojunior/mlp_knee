function generate_dynamic_data()
	data_loads = get_data_loads();
	training_set = clean_data(data_loads(1));


	save 'dynamics_data/training_set.mat' training_set;
    disp('Process completed successfully!');

end 
