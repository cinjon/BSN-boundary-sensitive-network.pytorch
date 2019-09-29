import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument('--module', type=str, default='TEM')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint')
    parser.add_argument('--checkpoint_epoch', type=int, default=None,
                        help="if none, use 'best'. else use that epoch.")    

    # Overall Dataset settings
    parser.add_argument('--dataset', default='gymnastics', type=str,
                        help='gymnsatics, thumosfeatures, thumosimages')
        
    parser.add_argument(
        '--video_info',
        type=str,
        default="./data/activitynet_annotations/video_info_new.csv")
    parser.add_argument(
        '--video_anno',
        type=str,
        default="./data/activitynet_annotations/anet_anno_action.json")

    # TEM Dataset settings
    parser.add_argument('--temporal_scale', type=int, default=100)
    parser.add_argument('--boundary_ratio', type=float, default=0.1)
    parser.add_argument('--feature_dirs', type=str, default=None,
                        help='comma delineated list of paths to feature directories')

    # PEM Dataset settings
    parser.add_argument('--pem_top_K', type=int, default=2500) # 500
    parser.add_argument('--pem_top_K_inference', type=int, default=2500) # 1000
    parser.add_argument('--pem_top_threshold', type=float, default=0,
                        help='instead of using pem_top_K, can do this to threshold the score and then use pem_top_K to randomly choose proposals from above this score.')    
    parser.add_argument('--pem_do_index', action='store_true')
    parser.add_argument('--pem_max_zero_weight', type=float, default=0.1)

    # TEM model settings
    parser.add_argument('--tem_feat_dim', type=int, default=400)
    parser.add_argument('--tem_hidden_dim', type=int, default=512)
    parser.add_argument('--tem_nonlinear_factor', type=float, default=0.01)
    parser.add_argument('--tem_reset_params', action='store_true')

    # PEM model settings
    parser.add_argument('--pem_feat_dim', type=int, default=32)
    parser.add_argument('--pem_hidden_dim', type=int, default=256)

    # TEM Training settings
    parser.add_argument('--tem_training_lr', type=float, default=0.001)
    parser.add_argument('--tem_weight_decay', type=float, default=0.0)
    parser.add_argument('--tem_l2_loss', type=float, default=0.005)
    parser.add_argument('--tem_epoch', type=int, default=30) # NOTE: was 20
    parser.add_argument('--tem_step_size', type=int, default=7)
    parser.add_argument('--tem_step_gamma', type=float, default=0.1) # 0.1
    parser.add_argument('--tem_lr_milestones', type=str, default='5') 
    parser.add_argument('--tem_batch_size', type=int, default=16)
    parser.add_argument('--tem_match_thres', type=float, default=0.5)
    parser.add_argument('--tem_compute_loss_interval', type=float, default=20)    
    parser.add_argument('--tem_train_subset', type=str, default='train', help='can be train or overfit.')
    parser.add_argument('--tem_results_dir', type=str, default=None, help='used for inference to generate the results that PGM_proposals uses.')
    parser.add_argument('--tem_results_subset', type=str, default='full', help='can be full, train, or overfit.')
    
    # PEM Training settings
    parser.add_argument('--pem_nonlinear_factor', type=int, default=0.1)
    parser.add_argument('--pem_training_lr', type=float, default=0.01)
    parser.add_argument('--pem_weight_decay', type=float, default=0.00001)
    parser.add_argument('--pem_l2_loss', type=float, default=0.000025)    
    parser.add_argument('--pem_epoch', type=int, default=20)
    parser.add_argument('--pem_step_size', type=int, default=10)
    parser.add_argument('--pem_step_gamma', type=float, default=0.1)
    parser.add_argument('--pem_batch_size', type=int, default=16)
    parser.add_argument('--pem_u_ratio_m', type=float, default=1)
    parser.add_argument('--pem_u_ratio_l', type=float, default=2)
    parser.add_argument('--pem_high_iou_thres', type=float, default=0.7)
    parser.add_argument('--pem_low_iou_thres', type=float, default=0.3)
    parser.add_argument('--pem_compute_loss_interval', type=float, default=20) 

    # PEM inference settings
    parser.add_argument('--pem_inference_results_dir', type=str, default=None, help='where to save the pem_inference results.')
    parser.add_argument('--pem_inference_subset',
                        type=str,
                        default="full")

    # PGM settings
    parser.add_argument('--pgm_threshold', type=float, default=0.5)
    parser.add_argument('--pgm_thread', type=int, default=8)
    # The original ahd it s.t. num_sample_start + end + action should equal to pem_feat_dim.
    # However, using the Thumos one, it appears to be num_sample_start*2 + num_sample_end*2 + action beacuse the action stuff is included in the first two as well...
    parser.add_argument('--num_sample_start', type=int, default=8)
    parser.add_argument('--num_sample_end', type=int, default=8)
    parser.add_argument(
        '--num_sample_action', type=int, default=16
    )  
    parser.add_argument('--num_sample_interpld', type=int, default=3)
    parser.add_argument('--bsp_boundary_ratio', type=float, default=0.2)
    parser.add_argument('--pgm_proposals_dir', type=str, default=None, help='used to save the pgm proposals.')
    parser.add_argument('--pgm_features_dir', type=str, default=None, help='used to save the pgm features.')
    parser.add_argument('--pgm_subset', type=str, default='full', help='can be full, train, or overfit.')
    parser.add_argument('--pgm_score_threshold', type=float, default=0.5)    
    
    # Post processing
    parser.add_argument('--post_process_top_K', type=int, default=100)
    parser.add_argument('--post_process_thread', type=int, default=8)
    parser.add_argument('--do_eval_after_postprocessing', action='store_true')    
    parser.add_argument('--soft_nms_alpha', type=float, default=0.75)
    parser.add_argument('--soft_nms_low_thres', type=float, default=0.65)
    parser.add_argument('--soft_nms_high_thres', type=float, default=0.9)
    parser.add_argument('--postprocessed_results_dir',
                        type=str,
                        default="/checkpoint/cinjon/spaceofmotion/bsn/postprocessing")
    parser.add_argument('--save_fig_path',
                        type=str,
                        default="./output/evaluation_result.jpg")

    parser.add_argument('--do_augment', action='store_true')
    parser.add_argument('--do_representation', action='store_true')
    parser.add_argument('--do_feat_conversion', action='store_true')
    parser.add_argument(
        '--representation_module',
        type=str,
        default='corrflow',
        help=
        'the underlying representation module when using one. should have a forward call that yields a frozen repr and a linear transform func that get sthat representation into a manageable size.'
    )
    parser.add_argument(
        '--representation_checkpoint',
        type=str,
        default=
        '/checkpoint/cinjon/spaceofmotion/supercons/corrflow.kineticsmodel.pth',
        help='the checkpoint for the underlying representation module.')
    parser.add_argument('--num_videoframes', type=int, default=100)
    parser.add_argument(
        '--skip_videoframes',
        type=int,
        default=5,
        help=
        'the number of video frames to skip in between each one. using 1 means that there is no skip.'
    )

    parser.add_argument('--log_to_comet', action='store_true', default=False)
    parser.add_argument('--log_to_comet_every', default=50, type=int)
    parser.add_argument('--local_comet_dir',
			type=str,
			default=None,
			help='local dir to process comet locally only. '
			'primarily for fb, will stop remote calls.')

    parser.add_argument('--name',
			type=str,
			help='the identifying name of this experiment.',
			default=None)
    parser.add_argument('--counter',
			type=int,
			help='the integer counter of this experiment. '
			'defaults to None because Cinjon is likely the '
			'only one who is going to use it.')
    parser.add_argument(
        '--data_workers',
        type=int,
        default=8,
        help='the number of workers to pull data',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='the seed',
    )
    parser.add_argument(
        '--num_gpus',
        type=int,
        default=1,
        help='the seed',
    )
    parser.add_argument(
        '--time',
        type=float,
        default=4,
        help='the number of hours',
    )
    
    args = parser.parse_args()

    return args
