import os
from os.path import join as p_join

scenes = ["room0", "room1", "room2", "office0", "office1", "office3",
          "office3", "office4", "apartment_0", "apartment_1",
          "apartment_2", "frl_apartment_0", "frl_apartment_4"]

primary_device="cuda:0"
seed = 6
scene_name = "office0"

map_every = 1
keyframe_every = 4
mapping_window_size = 24
tracking_iters = 40
mapping_iters = 60

group_name = "Replica"
run_name = f"{scene_name}_{seed}"

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=500, # Report Global Progress every nth frame
    eval_every=5, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=3, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=True, # Save Checkpoints
    checkpoint_interval=500, # Checkpoint Interval
    save_timestamp_keyframes=False,
    use_wandb=True,
    wandb=dict(
        entity="my-project", # Please change the entity name
        project="ESGS-SLAM",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        basedir="/home/6t/ZJJ/dataset/gsData/SemanicReplica/Replica",
        gradslam_data_cfg="./configs/data/replica.yaml",
        data_name= "replica",
        sequence=scene_name,
        desired_image_height=680,
        desired_image_width=1200,
        start=0,
        end=-1,
        stride=1,
        num_frames=-1, # Set to -1 to use all frames
        load_semantics=True,
        num_semantic_classes=101
    ),
    tracking=dict(
        use_gt_poses=False, # Use GT Poses for Tracking
        forward_prop=True, # Forward Propagate Poses
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            seg=0.05,
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.0004,
            cam_trans=0.002,
            semantic_colors=0.0,
        ),
    ),
    icp_params=dict(
        icp_damping = 0.0001,
        icp_downscale_iters = [ 5, 5, 5 ],
        icp_distance_threshold = 0.1 ,# m
        icp_downscales= [ 0.25, 0.5, 1.0 ],
        icp_fail_threshold= 0.02,
        icp_warmup_frames= 0,
        icp_use_model_depth= True, # if False, use dataset depth frame to frame
        icp_matches_threshold= 0.2, # ratio * valie pixels
        icp_normal_threshold= 20, # degree
        icp_sample_distance_threshold= 0.01, # m
        icp_sample_normal_threshold= 0.01, # cos similarity
        invalid_confidence_thresh= 0.2,
        use_gt_pose= False,
        verbose= False,
        min_depth= 0.3,
        max_depth= 5,
        depth_filter= False,
    )
    ,
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5, # For Addition of new Gaussians
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        first_frame_mapping_iters = 1000,
        opt_rskm_interval=5,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            seg=0.5,
            big_gaussian_reg=0.01,
            small_gaussian_reg=0.001,
        ),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
            semantic_colors=0.0025,
        ),
        prune_gaussians=True, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=500, # Doesn't consider iter 0
        ),
        use_gaussian_splatting_densification=False, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000, # Doesn't consider iter 0
        ),
    ),
    viz=dict(
        render_mode='color', # ['color', 'depth', 'centers', 'semantic_color']
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        visualize_cams=False, # Visualize Camera Frustums and Trajectory
        viz_w=600, viz_h=340,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5, # FPS for Online Recon Viz
        enter_interactive_post_online=True, # Enter Interactive Mode after Online Recon Viz
        scene_name=scene_name,
        load_semantics=True, # Whether load semantic information
    ),
    driod_tracking=dict(
        use_driod_tracking= True,
        pretrained = 'pretrained/droid.pth',
        buffer=400,
        only_tracking = False,
        max_age=50,
        warmup = 8,
        beta = 0.6,
        mono_thres=False,
        mono_prior=dict(
            depth = 'omnidata',
            depth_pretrained = 'pretrained/omnidata_dpt_depth_v2.ckpt',
            predict_online = True,
        ),
        motion_filter=dict(
            thresh=2.5,
        ),
        multiview_filter = dict(
            thresh = 0.01,
            visible_num = 2,
        ),
        frontend=dict(
            keyframe_thresh = 2.25,
            thresh = 25.0,
            window = 50,
            radius = 2,
            max_factors = 100,
            enable_online_ba= True,
            enable_loop = True,
            nms= 1,
        ),
        backend=dict(
            # used for loop detection
            final_ba =True,
            ba_freq = 50,
            thresh = 25.0,
            radius = 1,
            nms = 5,
            loop_window = 50,
            loop_thresh = 25.0,
            loop_radius = 1,
            loop_nms = 25,
            BA_type = 'DSPO',
            normalize = True,

        ),

    ),
    cam=dict(
        H_edge = 0,
        W_edge = 0,
        H_out = 320,
        W_out = 640,
        H = 680,
        W = 1200,
        fx = 600.0,
        fy = 600.0,
        cx = 599.5,
        cy= 339.5,
    )
)