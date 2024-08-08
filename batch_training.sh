for scene in apartment_0 apartment_1 apartment_2 frl_apartment_0 frl_apartment_1 frl_apartment_2 frl_apartment_3 frl_apartment_4 frl_apartment_5 hotel_0 office_0 office_1 office_2 office_3 office_4 room_0 room_1 room_2 ; do
    input_image_path=sample_data/replica/${scene}_lemniscate_1k_0/renders/input_rgb.jpg
    input_depth_path=sample_data/replica/${scene}_lemniscate_1k_0/renders/input_depth.dpt
    trajectory_file=model/trajectories/examples/lemniscate.json
    
    python generate_scene.py --input_image_path $input_image_path --input_depth_path $input_depth_path --trajectory_file $trajectory_file --exp_name $scene
    
    expname=output/lemniscate/${scene}

    python metrics.py --exp_renders_dir $expname --gt_renders_dir sample_data/replica/${scene}_lemniscate_1k_0/renders
done