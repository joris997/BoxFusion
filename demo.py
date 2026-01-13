import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import glob
import itertools
import numpy as np
import rerun
import rerun.blueprint as rrb
import yaml
import torch
import torchvision
import rclpy
import sys
import uuid
import open3d as o3d
from pathlib import Path
from PIL import Image
import rclpy 
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from tools.utils import * 
import open_clip 
import torch.nn.functional as F
import time

from boxfusion.cubify_transformer import make_cubify_transformer

from boxfusion.instances import Instances3D
from boxfusion.preprocessor import Augmentor, Preprocessor

from boxfusion.box_manager import BoxManager
from boxfusion.box_fusion import BoxFusion

from geometry_msgs.msg import PolygonStamped, Polygon, Point32

def text_to_feature(strings, clip_model, device="cuda"):
    tokenized = open_clip.tokenize(strings).to(device)
    with torch.no_grad():
        feats = clip_model.encode_text(tokenized)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


# def run(cfg, model, dataset, clip_model, preprocess, tokenized_text, text_features, augmentor, preprocessor, 
#         score_thresh=0.0, viz_on_gt_points=False, gap=25, re_vis=True, 
#         total_duration=60):
#     is_depth_model = "wide/depth" in augmentor.measurement_keys
#     blueprint = rrb.Blueprint(
#         rrb.Vertical(
#             contents=[
#                 rrb.Horizontal(
#                     contents=([
#                     rrb.Spatial3DView(
#                         name="World",
#                         contents=[
#                             "+ $origin/**",
#                             "+ /device/wide/pred_instances/**",
#                             # "+ /world/image/**"
#                         ],
#                         origin="/world"),
#                     ])),
#                 rrb.Horizontal(
#                     contents=([
#                         rrb.Spatial2DView(
#                             name="Image",
#                             origin="/device/wide/image",
#                             contents=[
#                                 "+ $origin/**",
#                                 "+ /device/wide/pred_instances/**"
#                             ])
#                     ] + ([
#                         # Only show this for RGB-D.
#                         rrb.Spatial2DView(
#                             name="Depth",
#                             origin="/device/wide/depth")
#                     ] if is_depth_model else [])),
#                     name="Wide")
#             ]))

#     recording = None
#     video_id = None

#     device = model.pixel_mean

#     count=0
#     all_pred_box = None
#     all_poses = None

#     all_kf_pose = {}
#     per_frame_ins = None #save every predicted boxes
#     traj_xyz = []

#     box_manager = BoxManager(cfg)
#     Box_Fuser = BoxFusion(cfg)

#     box_count = 0
#     start_time = time.time()
    
    
    
#     for sample in dataset:
#         sample_video_id = sample["meta"]["video_id"] #(['sensor_info', 'wide', 'gt', 'meta'])
#         pose = sample['sensor_info'].gt.RT
        
#         video_id = sample_video_id
#         if ((recording is None) or (video_id != sample_video_id)) and re_vis:
#             new_recording = rerun.new_recording(
#                 application_id=str(sample_video_id), recording_id=uuid.uuid4(), make_default=True)
#             new_recording.send_blueprint(blueprint, make_active=True)
#             rerun.spawn()
#             recording = new_recording
        
#         pose_np = pose.squeeze().cpu().numpy()

#         if re_vis:
#             rerun.set_time(timeline="pts", recording=recording, timestamp=sample["meta"]["timestamp"])

#         # -> channels last.
#         image = np.moveaxis(sample["wide"]["image"][-1].numpy(), 0, -1)  #[H,W,3]

#         if re_vis:
#             color_camera = rerun.Pinhole(
#                 image_from_camera=sample["sensor_info"].wide.image.K[-1].numpy(), resolution=sample["sensor_info"].wide.image.size)

#         if is_depth_model and re_vis:
#             # Show the depth being sent to the model.            
#             depth_camera = rerun.Pinhole(
#                 image_from_camera=sample["sensor_info"].wide.depth.K[-1].numpy(), resolution=sample["sensor_info"].wide.depth.size)

#         if Box_Fuser.update_K_flag == False:
#             Box_Fuser.update_intrinsics(sample["sensor_info"].wide.image.size,sample["sensor_info"].wide.image.K[-1].numpy()) #size:[W,H]

#         xyzrgb = None
#         if viz_on_gt_points and sample["sensor_info"].has("gt"):
#             # Backproject GT depth to world so we can compare our predictions.
#             depth_gt = sample["wide"]["depth"][-1]
#             matched_image = torch.tensor(np.array(Image.fromarray(image).resize((depth_gt.shape[1], depth_gt.shape[0]))))
#             # Feel free to change max_depth, but know CA is only trained up to 5m.
#             xyz, valid = unproject(depth_gt, sample["sensor_info"].gt.depth.K[-1], pose.squeeze(), max_depth=10.0)
#             xyzrgb = torch.cat((xyz, matched_image / 255.0), dim=-1)[valid]            
                    
#         packaged = augmentor.package(sample)
#         packaged = move_input_to_current_device(packaged, device)
#         packaged = preprocessor.preprocess([packaged])

#         # Every gap nth frame is selected as keyframe
#         if count % gap == 0:
#             with torch.no_grad():
#                 pred_instances = model(packaged)[0] 

#             pred_instances = pred_instances[pred_instances.scores >= float(score_thresh)]
 
#             if cfg["detection"]["uv_bound"]:
#                 uv_mask = box_manager.check_uv_bounds(pred_instances.pred_proj_xy,image.shape[1],image.shape[0],ratio=cfg["detection"]["uv_bound_value"]) #[N]
#                 pred_instances = pred_instances[uv_mask]
#             if cfg["detection"]["floor_mask"]:
#                 floor_mask = box_manager.check_floor_mask(pred_instances.pred_boxes_3d.tensor, ratio=cfg["detection"]["floor_ratio"])
#                 pred_instances = pred_instances[~floor_mask]

#            # avoid first frame empty predictions
#             if len(pred_instances) == 0 and count ==0:
#                 with torch.no_grad():
#                     pred_instances = model(packaged)[0]
#                 pred_instances = pred_instances[pred_instances.scores >= float(cfg['detection']['score_thresh']/4)]
#                 print("again",count,"pred_instances",len(pred_instances))
#                 if cfg["detection"]["uv_bound"]:
#                     uv_mask = box_manager.check_uv_bounds(pred_instances.pred_proj_xy,image.shape[1],image.shape[0],ratio=cfg["detection"]["uv_bound_value"]) #[N]
#                     pred_instances = pred_instances[uv_mask]
#                 print("again",count,"pred_instances",len(pred_instances))

#         # Hold off on logging anything until now, since the delay might confuse the user in the visualizer.
#         RT = sample["sensor_info"].gt.RT[-1].numpy()
#         if re_vis:
#             pose_transform = rerun.Transform3D(
#                 translation=RT[:3, 3],
#                 rotation=rerun.Quaternion(xyzw=Rotation.from_matrix(RT[:3, :3]).as_quat()))
#             rerun.log("/world/image", pose_transform)
#             rerun.log("/world/image", color_camera)

#             rerun.log("/device/wide/image", pose_transform)
#             rerun.log("/device/wide/image", rerun.Image(image).compress())
#             rerun.log("/device/wide/image", color_camera)
#         traj_xyz.append(RT[:3, 3])
            

#         if is_depth_model and re_vis:
#             rerun.log("/device/wide/depth", rerun.DepthImage(sample["wide"]["depth"][-1].numpy()))
#             rerun.log("/device/wide/depth", depth_camera)
        
#         if xyzrgb is not None and re_vis:
#             rerun.log("/world/xyz", rerun.Points3D(positions=xyzrgb[..., :3], colors=xyzrgb[..., 3:], radii=None))     

#         # visualize the trajectory
#         if cfg["vis"]["trajectory"] and re_vis:
#             rerun.log("/world/trajectory", rerun.LineStrips3D([np.array(traj_xyz)[:count]], colors=[84,255,159]))

#         # print(sample["sensor_info"].wide.depth.K[-1].numpy())
#         # only process keyframes
#         if count % gap ==0 or count == len(dataset)-1:
            
#             all_kf_pose[count] = pose_np
#             pose_np = np.expand_dims(pose_np,axis=0)
#             pose_np = np.repeat(pose_np, repeats=len(pred_instances), axis=0) 
            
#             if len(pred_instances)==0:
#                 all_pred_box = all_pred_box
#                 all_poses = all_poses
#                 box_count += len(pred_instances)
#                 box_manager.num_record[count] = box_count
#                 count+=1
#                 continue
            
#             # add new properties for Instance3D predictions
#             pred_instances.categories = np.array(['None'] * len(pred_instances)) # Initialize category labels as 'None' for all predicted instances
#             pred_instances.cam_pose = torch.from_numpy(pose_np) # Convert camera pose from numpy to tensor and assign to instances
#             pred_instances.frame_id = torch.tensor([count]).repeat(pose_np.shape[0]) # Assign current frame ID to all instances in this frame
#             pred_instances.init_id = box_count+torch.arange(len(pred_instances)) # Create unique initial IDs for each instance based on global box count
#             pred_instances.valid_num = torch.zeros(len(pred_instances)) # Initialize validation counter to zero for all instances
#             pred_instances.pred_boxes_3d.transform2world(pred_instances.cam_pose) # Transform 3D bounding boxes from camera coordinates to world coordinates
#             pred_instances.project_3d_boxes(sample["sensor_info"].wide.depth.K[-1].numpy(), H=image.shape[0],W=image.shape[1]) # Project 3D boxes to 2D image coordinates using camera intrinsics

#             # record how many boxes each keyframe has, so we know which box belongs to which frame
#             box_count += len(pred_instances)
#             box_manager.num_record[count] = box_count
 
#             # first keyframe, initialize some data structures
#             if all_pred_box is None and count<gap:
                
#                 #predict the semantic classes
#                 boxes = pred_instances.pred_boxes.cpu().numpy()
#                 #scale the boxes by
#                 boxes = scale_boxes(boxes,image.shape[0],image.shape[1],scale=cfg['detection']['scale_box'])

#                 class_results, box_features = text_prompt(boxes, tokenized_text, text_features, image, clip_model, preprocess) #[N_box]
#                 pred_instances.categories = class_results
#                 print("frame",count," initial class_results:",class_results)

#                 all_pred_box = pred_instances
#                 all_poses = pose_np
#                 per_frame_ins = pred_instances
 
#                 #record the current frame boxes info
#                 box_manager.init_new_predictions(len(pred_instances),0)

#             else:
                
#                 box_manager.init_new_predictions(len(pred_instances),len(per_frame_ins))

#                 num_before_cat = len(all_pred_box)
#                 cur_global_pred_box = all_pred_box

#                 all_pred_box = Instances3D.cat([all_pred_box,pred_instances])
#                 per_frame_ins = Instances3D.cat([per_frame_ins,pred_instances])

#                 all_poses = np.concatenate((all_poses, pose_np), axis=0)  

#                 print("\ncur frame id:",count)
#                 '''
#                 STEP1: spatial association using 3D OBB NMS
#                 '''
#                 mask, success_mask = Instances3D.spatial_association(all_pred_box,cfg["box_fusion"]["nms_threshold"],box_manager,per_frame_ins.cam_pose)
                
#                 cur_keep_idx = [i-num_before_cat for i in mask if i>=num_before_cat]
#                 cur_success_nms = [i-num_before_cat for i in success_mask if i>=num_before_cat]
                
 
#                 keep_idx = np.asarray(mask)
#                 if len(cur_keep_idx)>0:
#                     '''
#                     STEP2: correspondence association for small objects
#                     '''
#                     all_pred_box,all_poses,keep_idx = Instances3D.correspondence_association(
#                         cfg, 
#                         box_manager, 
#                         cur_keep_idx, 
#                         cur_success_nms,
#                         pred_instances, 
#                         cur_global_pred_box, 
#                         all_pred_box,all_poses, 
#                         per_frame_ins.cam_pose, 
#                         count,
#                         mask,
#                         sample["sensor_info"].gt.depth.K[-1],
#                         all_kf_pose,
#                         threshold=cfg['association']['small_threshold'],
#                         H=image.shape[0],
#                         W=image.shape[1]
#                         )

#                     # update the fusion list based on keep_idx
#                     box_manager.update(keep_idx)
                
#                     print(count," box_manager",box_manager.fusion_list)

#                     #filter those evident wrong boxes that valid_num=0
#                     if cfg['box_fusion']['check_valid']:
#                         all_pred_box = box_manager.check_valid_num(all_pred_box, count, gap)

#                     '''
#                     multi-view box fusion
#                     '''
#                     print("frame_id:box_num",box_manager.num_record)
#                     if cfg['box_fusion']['use']:
#                         Box_Fuser.boxfusion(all_pred_box, per_frame_ins, box_manager)
                
#                     #predict the semantic classes of remaining new boxes
#                     cur_keep_idx = [i-num_before_cat for i in keep_idx if i>=num_before_cat]
#                     cur_keep_idx_in_all = [i for i in range(keep_idx.shape[0]) if keep_idx[i]>=num_before_cat]

#                     if len(cur_keep_idx)>0:
#                         boxes = pred_instances.pred_boxes.cpu().numpy()
#                         boxes = boxes[cur_keep_idx]
#                         # scale the boxes
#                         boxes = scale_boxes(boxes,image.shape[0],image.shape[1],scale=cfg['detection']['scale_box'])
#                         # if len(pred_instances)>0:
#                         class_results, box_features = text_prompt(boxes, tokenized_text, text_features, image, clip_model, preprocess) #[N_box]
#                         all_pred_box.categories[cur_keep_idx_in_all] = class_results
#                         print("frame",count," new box class_results:",class_results)

#                 else: # no new box
#                     all_pred_box = all_pred_box[mask]
#                     all_poses = all_poses[mask]
#                     box_manager.update(keep_idx)
#                     print(count, "new boxes have all been nms"," box_manager",box_manager.fusion_list)

#             if re_vis:
#                 visualize_online_boxes(all_pred_box, prefix="/device/wide", boxes_3d_name="pred_boxes_3d", log_instances_name="pred_instances",count=count,save=False,show_class=cfg["vis"]["show_class"],show_label=cfg["vis"]["show_label"]) 

#         count+=1
#         elapsed_time = time.time() - start_time
#         if elapsed_time >= total_duration:
#             print(f"Reached total duration of {total_duration} seconds. Stopping capture.")
#             rclpy.shutdown()
#             break
        
#     # # save the results
#     # if count == len(dataset)-1 or (count+gap)>len(dataset)-1:
#     end_time = time.time()
#     duration = end_time - start_time  
#     fps = count / duration
#     print(f"Cost: {duration:.2f} s", f"Average FPS: {fps:.2f}")
    
#     # save global boxes for evaluation
#     if cfg['data']['output_dir'] is not None:# and cfg["eval"]:
#         class_list = tokenized_text.tolist()
#         class_idx = np.array([class_list.index(c) for c in all_pred_box.categories]) #[N]

#         boxes_3d = all_pred_box.pred_boxes_3d.corners.cpu().numpy() # [N,8,3]
#         if cfg['dataset'] == 'scannet':
#             boxes_3d = post_process(boxes_3d)
            
#         if boxes_3d.shape[0]>0:
#             save_list = [[(int(n), (boxes_3d[n]), class_list[class_idx[n]]) for n in range(len(all_pred_box))]] # list of tuples class_idx[n]
#             save_box(save_list, os.path.join(cfg['data']['output_dir'], video_id[0]+"_boxes.pkl"))

#         # save the pointcloud, xyzrgb, (numpy object) to pkl
#         torch.save(xyzrgb.cpu(), os.path.join(cfg['data']['output_dir'], video_id[0]+'_xyzrgb.pt'))
        
#     # publish the first box
#     if all_pred_box is not None and len(all_pred_box)>0:
#         first_box = all_pred_box.pred_boxes_3d.corners[0].cpu().numpy()  #[8,3]
#         publisher_node.publish_boxes(first_box)
#     return
    # exit(0)
    # break

# small ros node that publishes the online 3d detection boxes 
class Online3DNode(Node):
    def __init__(self):
        super().__init__('online_3d_node')

        # create publisher as list of PolygonStamped
        self.box_publisher = self.create_publisher(PolygonStamped, 'online_3d_boxes', 10)

        args = {'dataset_path': 'online',
                'model_path': './models/cutr_rgbd.pth',
                'config': './config/online.yaml',
                'clip_path': './models/open_clip_pytorch_model.bin',
                'seq': 'None',
                'class_txt': './data/panoptic_categories_nomerge.txt',
                'every_nth_frame': None,
                'viz_on_gt_points': True,
                'device': 'cuda',
                'video_ids': None,
                'primed_strings': None,
                'total_duration': 10}

        dataset_path = args['dataset_path']
        use_cache = False
        
        if dataset_path.lower() in ["scannet", "ca1m", 'online']:
            if not os.path.exists(args['config']):
                raise ValueError("Missing config path")
            else:
                with open(args['config'], 'r') as  f:
                    cfg = yaml.full_load(f)
            # load the customized sequence if given by the user
            if args['seq'] is not None:
                if dataset_path.lower()=='ca1m':
                    if 'example' in cfg['data']['datadir']:
                        current_file_path = os.path.abspath(__file__)
                        current_dir = os.path.dirname(current_file_path)
                        cfg['data']['datadir'] = os.path.join(current_dir, cfg['data']['datadir'])

                    else:
                        new_datadir = os.path.join(os.path.dirname(os.path.dirname(cfg['data']['datadir'])),  args['seq']+'/')
                        cfg['data']['datadir'] = new_datadir

                    
                else:
                    new_datadir = os.path.join(os.path.dirname(os.path.dirname(cfg['data']['datadir'])),  args['seq']+'/frames/')
                    cfg['data']['datadir'] = new_datadir
                    
                # eval only
                if os.path.exists(os.path.join(cfg['data']['output_dir'],args['seq']+"_boxes.pkl")) and cfg["eval"]:
                    print("Results for boxes already exist, skip evaluation")
                    sys.exit(0)
            
            dataset = get_dataset(cfg)

        assert args['model_path'] is not None
        checkpoint = torch.load(args['model_path'], map_location=args['device'] or "cpu")["model"]
        backbone_embedding_dimension = checkpoint["backbone.0.patch_embed.proj.weight"].shape[0]
            
        is_depth_model = True 
        model = make_cubify_transformer(dimension=backbone_embedding_dimension, depth_model=is_depth_model).eval()
        model.load_state_dict(checkpoint)

        dataset.load_arkit_depth = True
        if args['every_nth_frame'] is not None:
            dataset = itertools.islice(dataset, 0, None, args['every_nth_frame'])

        augmentor = Augmentor(("wide/image", "wide/depth"))
        preprocessor = Preprocessor()
        
        if args['device'] is not None:
            model = model.to(args['device'])
            clip_model, preprocess = load_clip(args['clip_path'])
            text_class = np.genfromtxt(args['class_txt'], delimiter='\n', dtype=str) 
            print(f"text classes {text_class}")
            text_features = torch.load('./data/class_features.pt').cuda()

            extra_strings = ['tangerine', 'cup']
            # append extra_strings to text_class
            text_class = np.concatenate((text_class, np.array(extra_strings)), axis=0)
            # append extra_strings features to text_features
            extra_features = text_to_feature(extra_strings, clip_model, device=args['device'])
            text_features = torch.cat((text_features, extra_features), dim=0)
            # print(f"my embedding = their embedding?: {text_features[0,:] == text_to_feature([text_class[0]], clip_model, device=args.device)[0,:]}")
            # print("Loaded text features:", text_features.shape)
        
        self.run(cfg, model, dataset, clip_model, preprocess, text_class, text_features, augmentor, preprocessor, 
            score_thresh=cfg['detection']['score_thresh'], viz_on_gt_points=args['viz_on_gt_points'], gap=25, re_vis=True, 
            total_duration=args['total_duration']
        )

    def publish_boxes(self, box):
        msg = PolygonStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"

        polygon = Polygon()
        for corner in box:
            point = Point32()
            print(corner)
            point.x = float(corner[0])
            point.y = float(corner[1])
            point.z = float(corner[2])
            polygon.points.append(point)

        msg.polygon = polygon
        self.box_publisher.publish(msg)
    
    def run(self, 
        cfg, model, dataset, clip_model, preprocess, tokenized_text, text_features, augmentor, preprocessor, 
        score_thresh=0.0, viz_on_gt_points=False, gap=25, re_vis=True, 
        total_duration=60):
        is_depth_model = "wide/depth" in augmentor.measurement_keys
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                contents=[
                    rrb.Horizontal(
                        contents=([
                        rrb.Spatial3DView(
                            name="World",
                            contents=[
                                "+ $origin/**",
                                "+ /device/wide/pred_instances/**",
                                # "+ /world/image/**"
                            ],
                            origin="/world"),
                        ])),
                    rrb.Horizontal(
                        contents=([
                            rrb.Spatial2DView(
                                name="Image",
                                origin="/device/wide/image",
                                contents=[
                                    "+ $origin/**",
                                    "+ /device/wide/pred_instances/**"
                                ])
                        ] + ([
                            # Only show this for RGB-D.
                            rrb.Spatial2DView(
                                name="Depth",
                                origin="/device/wide/depth")
                        ] if is_depth_model else [])),
                        name="Wide")
                ]))

        recording = None
        video_id = None

        device = model.pixel_mean

        count=0
        all_pred_box = None
        all_poses = None

        all_kf_pose = {}
        per_frame_ins = None #save every predicted boxes
        traj_xyz = []

        box_manager = BoxManager(cfg)
        Box_Fuser = BoxFusion(cfg)

        box_count = 0
        start_time = time.time()
        
        
        
        for sample in dataset:
            sample_video_id = sample["meta"]["video_id"] #(['sensor_info', 'wide', 'gt', 'meta'])
            pose = sample['sensor_info'].gt.RT
            
            video_id = sample_video_id
            if ((recording is None) or (video_id != sample_video_id)) and re_vis:
                new_recording = rerun.new_recording(
                    application_id=str(sample_video_id), recording_id=uuid.uuid4(), make_default=True)
                new_recording.send_blueprint(blueprint, make_active=True)
                rerun.spawn()
                recording = new_recording
            
            pose_np = pose.squeeze().cpu().numpy()

            if re_vis:
                rerun.set_time(timeline="pts", recording=recording, timestamp=sample["meta"]["timestamp"])

            # -> channels last.
            image = np.moveaxis(sample["wide"]["image"][-1].numpy(), 0, -1)  #[H,W,3]

            if re_vis:
                color_camera = rerun.Pinhole(
                    image_from_camera=sample["sensor_info"].wide.image.K[-1].numpy(), resolution=sample["sensor_info"].wide.image.size)

            if is_depth_model and re_vis:
                # Show the depth being sent to the model.            
                depth_camera = rerun.Pinhole(
                    image_from_camera=sample["sensor_info"].wide.depth.K[-1].numpy(), resolution=sample["sensor_info"].wide.depth.size)

            if Box_Fuser.update_K_flag == False:
                Box_Fuser.update_intrinsics(sample["sensor_info"].wide.image.size,sample["sensor_info"].wide.image.K[-1].numpy()) #size:[W,H]

            xyzrgb = None
            if viz_on_gt_points and sample["sensor_info"].has("gt"):
                # Backproject GT depth to world so we can compare our predictions.
                depth_gt = sample["wide"]["depth"][-1]
                matched_image = torch.tensor(np.array(Image.fromarray(image).resize((depth_gt.shape[1], depth_gt.shape[0]))))
                # Feel free to change max_depth, but know CA is only trained up to 5m.
                xyz, valid = unproject(depth_gt, sample["sensor_info"].gt.depth.K[-1], pose.squeeze(), max_depth=10.0)
                xyzrgb = torch.cat((xyz, matched_image / 255.0), dim=-1)[valid]            
                        
            packaged = augmentor.package(sample)
            packaged = move_input_to_current_device(packaged, device)
            packaged = preprocessor.preprocess([packaged])

            # Every gap nth frame is selected as keyframe
            if count % gap == 0:
                with torch.no_grad():
                    pred_instances = model(packaged)[0] 

                pred_instances = pred_instances[pred_instances.scores >= float(score_thresh)]
    
                if cfg["detection"]["uv_bound"]:
                    uv_mask = box_manager.check_uv_bounds(pred_instances.pred_proj_xy,image.shape[1],image.shape[0],ratio=cfg["detection"]["uv_bound_value"]) #[N]
                    pred_instances = pred_instances[uv_mask]
                if cfg["detection"]["floor_mask"]:
                    floor_mask = box_manager.check_floor_mask(pred_instances.pred_boxes_3d.tensor, ratio=cfg["detection"]["floor_ratio"])
                    pred_instances = pred_instances[~floor_mask]

            # avoid first frame empty predictions
                if len(pred_instances) == 0 and count ==0:
                    with torch.no_grad():
                        pred_instances = model(packaged)[0]
                    pred_instances = pred_instances[pred_instances.scores >= float(cfg['detection']['score_thresh']/4)]
                    print("again",count,"pred_instances",len(pred_instances))
                    if cfg["detection"]["uv_bound"]:
                        uv_mask = box_manager.check_uv_bounds(pred_instances.pred_proj_xy,image.shape[1],image.shape[0],ratio=cfg["detection"]["uv_bound_value"]) #[N]
                        pred_instances = pred_instances[uv_mask]
                    print("again",count,"pred_instances",len(pred_instances))

            # Hold off on logging anything until now, since the delay might confuse the user in the visualizer.
            RT = sample["sensor_info"].gt.RT[-1].numpy()
            if re_vis:
                pose_transform = rerun.Transform3D(
                    translation=RT[:3, 3],
                    rotation=rerun.Quaternion(xyzw=Rotation.from_matrix(RT[:3, :3]).as_quat()))
                rerun.log("/world/image", pose_transform)
                rerun.log("/world/image", color_camera)

                rerun.log("/device/wide/image", pose_transform)
                rerun.log("/device/wide/image", rerun.Image(image).compress())
                rerun.log("/device/wide/image", color_camera)
            traj_xyz.append(RT[:3, 3])
                

            if is_depth_model and re_vis:
                rerun.log("/device/wide/depth", rerun.DepthImage(sample["wide"]["depth"][-1].numpy()))
                rerun.log("/device/wide/depth", depth_camera)
            
            if xyzrgb is not None and re_vis:
                rerun.log("/world/xyz", rerun.Points3D(positions=xyzrgb[..., :3], colors=xyzrgb[..., 3:], radii=None))     

            # visualize the trajectory
            if cfg["vis"]["trajectory"] and re_vis:
                rerun.log("/world/trajectory", rerun.LineStrips3D([np.array(traj_xyz)[:count]], colors=[84,255,159]))

            # print(sample["sensor_info"].wide.depth.K[-1].numpy())
            # only process keyframes
            if count % gap ==0 or count == len(dataset)-1:
                
                all_kf_pose[count] = pose_np
                pose_np = np.expand_dims(pose_np,axis=0)
                pose_np = np.repeat(pose_np, repeats=len(pred_instances), axis=0) 
                
                if len(pred_instances)==0:
                    all_pred_box = all_pred_box
                    all_poses = all_poses
                    box_count += len(pred_instances)
                    box_manager.num_record[count] = box_count
                    count+=1
                    continue
                
                # add new properties for Instance3D predictions
                pred_instances.categories = np.array(['None'] * len(pred_instances)) # Initialize category labels as 'None' for all predicted instances
                pred_instances.cam_pose = torch.from_numpy(pose_np) # Convert camera pose from numpy to tensor and assign to instances
                pred_instances.frame_id = torch.tensor([count]).repeat(pose_np.shape[0]) # Assign current frame ID to all instances in this frame
                pred_instances.init_id = box_count+torch.arange(len(pred_instances)) # Create unique initial IDs for each instance based on global box count
                pred_instances.valid_num = torch.zeros(len(pred_instances)) # Initialize validation counter to zero for all instances
                pred_instances.pred_boxes_3d.transform2world(pred_instances.cam_pose) # Transform 3D bounding boxes from camera coordinates to world coordinates
                pred_instances.project_3d_boxes(sample["sensor_info"].wide.depth.K[-1].numpy(), H=image.shape[0],W=image.shape[1]) # Project 3D boxes to 2D image coordinates using camera intrinsics

                # record how many boxes each keyframe has, so we know which box belongs to which frame
                box_count += len(pred_instances)
                box_manager.num_record[count] = box_count
    
                # first keyframe, initialize some data structures
                if all_pred_box is None and count<gap:
                    
                    #predict the semantic classes
                    boxes = pred_instances.pred_boxes.cpu().numpy()
                    #scale the boxes by
                    boxes = scale_boxes(boxes,image.shape[0],image.shape[1],scale=cfg['detection']['scale_box'])

                    class_results, box_features = text_prompt(boxes, tokenized_text, text_features, image, clip_model, preprocess) #[N_box]
                    pred_instances.categories = class_results
                    print("frame",count," initial class_results:",class_results)

                    all_pred_box = pred_instances
                    all_poses = pose_np
                    per_frame_ins = pred_instances
    
                    #record the current frame boxes info
                    box_manager.init_new_predictions(len(pred_instances),0)

                else:
                    
                    box_manager.init_new_predictions(len(pred_instances),len(per_frame_ins))

                    num_before_cat = len(all_pred_box)
                    cur_global_pred_box = all_pred_box

                    all_pred_box = Instances3D.cat([all_pred_box,pred_instances])
                    per_frame_ins = Instances3D.cat([per_frame_ins,pred_instances])

                    all_poses = np.concatenate((all_poses, pose_np), axis=0)  

                    print("\ncur frame id:",count)
                    '''
                    STEP1: spatial association using 3D OBB NMS
                    '''
                    mask, success_mask = Instances3D.spatial_association(all_pred_box,cfg["box_fusion"]["nms_threshold"],box_manager,per_frame_ins.cam_pose)
                    
                    cur_keep_idx = [i-num_before_cat for i in mask if i>=num_before_cat]
                    cur_success_nms = [i-num_before_cat for i in success_mask if i>=num_before_cat]
                    
    
                    keep_idx = np.asarray(mask)
                    if len(cur_keep_idx)>0:
                        '''
                        STEP2: correspondence association for small objects
                        '''
                        all_pred_box,all_poses,keep_idx = Instances3D.correspondence_association(
                            cfg, 
                            box_manager, 
                            cur_keep_idx, 
                            cur_success_nms,
                            pred_instances, 
                            cur_global_pred_box, 
                            all_pred_box,all_poses, 
                            per_frame_ins.cam_pose, 
                            count,
                            mask,
                            sample["sensor_info"].gt.depth.K[-1],
                            all_kf_pose,
                            threshold=cfg['association']['small_threshold'],
                            H=image.shape[0],
                            W=image.shape[1]
                            )

                        # update the fusion list based on keep_idx
                        box_manager.update(keep_idx)
                    
                        print(count," box_manager",box_manager.fusion_list)

                        #filter those evident wrong boxes that valid_num=0
                        if cfg['box_fusion']['check_valid']:
                            all_pred_box = box_manager.check_valid_num(all_pred_box, count, gap)

                        '''
                        multi-view box fusion
                        '''
                        print("frame_id:box_num",box_manager.num_record)
                        if cfg['box_fusion']['use']:
                            Box_Fuser.boxfusion(all_pred_box, per_frame_ins, box_manager)
                    
                        #predict the semantic classes of remaining new boxes
                        cur_keep_idx = [i-num_before_cat for i in keep_idx if i>=num_before_cat]
                        cur_keep_idx_in_all = [i for i in range(keep_idx.shape[0]) if keep_idx[i]>=num_before_cat]

                        if len(cur_keep_idx)>0:
                            boxes = pred_instances.pred_boxes.cpu().numpy()
                            boxes = boxes[cur_keep_idx]
                            # scale the boxes
                            boxes = scale_boxes(boxes,image.shape[0],image.shape[1],scale=cfg['detection']['scale_box'])
                            # if len(pred_instances)>0:
                            class_results, box_features = text_prompt(boxes, tokenized_text, text_features, image, clip_model, preprocess) #[N_box]
                            all_pred_box.categories[cur_keep_idx_in_all] = class_results
                            print("frame",count," new box class_results:",class_results)

                    else: # no new box
                        all_pred_box = all_pred_box[mask]
                        all_poses = all_poses[mask]
                        box_manager.update(keep_idx)
                        print(count, "new boxes have all been nms"," box_manager",box_manager.fusion_list)

                if re_vis:
                    visualize_online_boxes(all_pred_box, prefix="/device/wide", boxes_3d_name="pred_boxes_3d", log_instances_name="pred_instances",count=count,save=False,show_class=cfg["vis"]["show_class"],show_label=cfg["vis"]["show_label"]) 

            count+=1
            elapsed_time = time.time() - start_time
            if elapsed_time >= total_duration:
                print(f"Reached total duration of {total_duration} seconds. Stopping capture.")
                rclpy.shutdown()
                break
            
        # # save the results
        # if count == len(dataset)-1 or (count+gap)>len(dataset)-1:
        end_time = time.time()
        duration = end_time - start_time  
        fps = count / duration
        print(f"Cost: {duration:.2f} s", f"Average FPS: {fps:.2f}")
        
        # save global boxes for evaluation
        if cfg['data']['output_dir'] is not None:# and cfg["eval"]:
            class_list = tokenized_text.tolist()
            class_idx = np.array([class_list.index(c) for c in all_pred_box.categories]) #[N]

            boxes_3d = all_pred_box.pred_boxes_3d.corners.cpu().numpy() # [N,8,3]
            if cfg['dataset'] == 'scannet':
                boxes_3d = post_process(boxes_3d)
                
            if boxes_3d.shape[0]>0:
                save_list = [[(int(n), (boxes_3d[n]), class_list[class_idx[n]]) for n in range(len(all_pred_box))]] # list of tuples class_idx[n]
                save_box(save_list, os.path.join(cfg['data']['output_dir'], video_id[0]+"_boxes.pkl"))

            # save the pointcloud, xyzrgb, (numpy object) to pkl
            torch.save(xyzrgb.cpu(), os.path.join(cfg['data']['output_dir'], video_id[0]+'_xyzrgb.pt'))
            
        # publish the first box
        if all_pred_box is not None and len(all_pred_box)>0:
            first_box = all_pred_box.pred_boxes_3d.corners[0].cpu().numpy()  #[8,3]
            self.publish_boxes(first_box)
        return

if __name__ == "__main__":
    rclpy.init()
    node = Online3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down online 3D node")
        node.destroy_node()
        rclpy.shutdown()


# class Online3DNode(Node):
#     def __init__(self):
#         super().__init__('online_3d_node')

#         # Parse arguments
#         parser = argparse.ArgumentParser()
#         parser.add_argument("--dataset_path", default='online', help="Path to dataset or stream")
#         parser.add_argument("--model_path", default='./models/cutr_rgbd.pth', help="Path to model checkpoint")
#         parser.add_argument("--config", default='./config/online.yaml', type=str)
#         parser.add_argument("--clip_path", default='./models/open_clip_pytorch_model.bin', type=str)
#         parser.add_argument("--seq", default='None', type=str)
#         parser.add_argument("--class_txt", default='./data/panoptic_categories_nomerge.txt', type=str)
#         parser.add_argument("--every-nth-frame", default=None, type=int)
#         parser.add_argument("--viz-on-gt-points", action="store_true", default=True)
#         parser.add_argument("--device", default="cuda")
#         parser.add_argument("--total_duration", type=int, default=10)
#         args = parser.parse_args(sys.argv[1:])
#         self.args = args

#         # Load config
#         if not os.path.exists(args.config):
#             self.get_logger().error(f"Config path {args.config} not found")
#             sys.exit(1)
#         with open(args.config, 'r') as f:
#             self.cfg = yaml.full_load(f)

#         # Load dataset
#         self.dataset = get_dataset(self.cfg)
#         if args.every_nth_frame is not None:
#             self.dataset = itertools.islice(self.dataset, 0, None, args.every_nth_frame)

#         # Load model
#         checkpoint = torch.load(args.model_path, map_location=args.device or "cpu")["model"]
#         backbone_embedding_dimension = checkpoint["backbone.0.patch_embed.proj.weight"].shape[0]
#         self.model = make_cubify_transformer(dimension=backbone_embedding_dimension, depth_model=True).eval()
#         self.model.load_state_dict(checkpoint)
#         self.model = self.model.to(args.device)

#         # Publisher for boxes (with labels)
#         self.box_publisher = self.create_publisher(Boxes3DStamped, 'predicted_boxes', 10)

#         # Load CLIP
#         self.clip_model, self.preprocess = load_clip(args.clip_path)
#         text_class = np.genfromtxt(args.class_txt, delimiter='\n', dtype=str)
#         text_features = torch.load('./data/class_features.pt').cuda()
#         self.text_class = text_class
#         self.text_features = text_features

#         # Augmentor / Preprocessor
#         self.augmentor = Augmentor(("wide/image", "wide/depth"))
#         self.preprocessor = Preprocessor()

#         self.get_logger().info("Online 3D node initialized, starting run...")
#         self.start_run()

#     def start_run(self):
#         run(self.cfg,
#             self.model,
#             self.dataset,
#             self.clip_model,
#             self.preprocess,
#             self.text_class,
#             self.text_features,
#             self.augmentor,
#             self.preprocessor,
#             score_thresh=self.cfg['detection']['score_thresh'],
#             viz_on_gt_points=self.args.viz_on_gt_points,
#             gap=self.cfg["data"]["gap"],
#             re_vis=self.cfg['vis']['rerun'],
#             total_duration=self.args.total_duration)

# def main(args=None):
#     rclpy.init(args=args)
#     node = Online3DNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.get_logger().info("Shutting down online 3D node")
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == "__main__":
#     main()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--dataset_path", default='online',                               help="Path to the directory containing the .tar files, the full path to a single tar file (recommended), or a path to a txt file containing HTTP links. Using the value \"stream\" will attempt to stream from your device using the NeRFCapture app")
#     parser.add_argument("--model-path", default='./models/cutr_rgbd.pth',               help="Path to the model to load")
#     parser.add_argument("--config",     default='./config/online.yaml',                 type=str, help="config_path")
#     parser.add_argument("--clip_path",  default='./models/open_clip_pytorch_model.bin', type=str, help="Path to the CLIP model")
#     parser.add_argument("--seq",        default='None',                                 type=str, help="config_path")
#     parser.add_argument("--class_txt",  default='./data/panoptic_categories_nomerge.txt', type=str, help="config_path")
#     parser.add_argument("--every-nth-frame",    default=None,                           type =int, help="Load every `n` frames")
#     parser.add_argument("--viz-on-gt-points",   default=True,                           action="store_true", help="Backproject the GT depth to form a point cloud in order to visualize the predictions")
#     parser.add_argument("--device",             default="cuda",                         help="Which device to push the model to (cpu, mps, cuda)")
#     parser.add_argument("--video-ids",          nargs="+",                              help="Subset of videos to execute on. By default, all. Ignored if a tar file is explicitly given or in stream mode.")
#     parser.add_argument("--primed-strings",     default=None, nargs='*',                help="Extra CLIP text strings to prime the model with (e.g. detect 'green apple' instead of 'apple')")
#     parser.add_argument("--total_duration",      default=10,                             type=int, help="total duration to capture from stream")

#     args = parser.parse_args()
#     print("Command Line Args:", args)

#     dataset_path = args.dataset_path
#     use_cache = False
    
#     if dataset_path.lower() in ["scannet", "ca1m", 'online']:
#         if not os.path.exists(args.config):
#             raise ValueError("Missing config path")
#         else:
#             with open(args.config, 'r') as  f:
#                 cfg = yaml.full_load(f)
#         # load the customized sequence if given by the user
#         if args.seq is not None:
#             if dataset_path.lower()=='ca1m':
#                 if 'example' in cfg['data']['datadir']:
#                     current_file_path = os.path.abspath(__file__)
#                     current_dir = os.path.dirname(current_file_path)
#                     cfg['data']['datadir'] = os.path.join(current_dir, cfg['data']['datadir'])

#                 else:
#                     new_datadir = os.path.join(os.path.dirname(os.path.dirname(cfg['data']['datadir'])),  args.seq+'/')
#                     cfg['data']['datadir'] = new_datadir

                
#             else:
#                 new_datadir = os.path.join(os.path.dirname(os.path.dirname(cfg['data']['datadir'])),  args.seq+'/frames/')
#                 cfg['data']['datadir'] = new_datadir
                
#             # eval only
#             if os.path.exists(os.path.join(cfg['data']['output_dir'],args.seq+"_boxes.pkl")) and cfg["eval"]:
#                 print("Results for boxes already exist, skip evaluation")
#                 sys.exit(0)
        
#         dataset = get_dataset(cfg)

#     assert args.model_path is not None
#     checkpoint = torch.load(args.model_path, map_location=args.device or "cpu")["model"]
#     backbone_embedding_dimension = checkpoint["backbone.0.patch_embed.proj.weight"].shape[0]
        
#     is_depth_model = True 
#     model = make_cubify_transformer(dimension=backbone_embedding_dimension, depth_model=is_depth_model).eval()
#     model.load_state_dict(checkpoint)

#     dataset.load_arkit_depth = True
#     if args.every_nth_frame is not None:
#         dataset = itertools.islice(dataset, 0, None, args.every_nth_frame)

#     augmentor = Augmentor(("wide/image", "wide/depth"))
#     preprocessor = Preprocessor()
    
#     if args.device is not None:
#         model = model.to(args.device)
#         clip_model, preprocess = load_clip(args.clip_path)
#         text_class = np.genfromtxt(args.class_txt, delimiter='\n', dtype=str) 
#         print(f"text classes {text_class}")
#         text_features = torch.load('./data/class_features.pt').cuda()

#         extra_strings = ['tangerine', 'cup']
#         # append extra_strings to text_class
#         text_class = np.concatenate((text_class, np.array(extra_strings)), axis=0)
#         # append extra_strings features to text_features
#         extra_features = text_to_feature(extra_strings, clip_model, device=args.device)
#         text_features = torch.cat((text_features, extra_features), dim=0)
#         # print(f"my embedding = their embedding?: {text_features[0,:] == text_to_feature([text_class[0]], clip_model, device=args.device)[0,:]}")
#         # print("Loaded text features:", text_features.shape)

#     run(cfg, model, dataset, clip_model, preprocess, text_class, text_features, augmentor, preprocessor, 
#         score_thresh=cfg['detection']['score_thresh'], viz_on_gt_points=args.viz_on_gt_points, gap=cfg["data"]["gap"], re_vis=cfg['vis']['rerun'],
#         total_duration=args.total_duration)