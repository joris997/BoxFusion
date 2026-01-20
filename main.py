#!/home/none/venvs/boxfusion/bin/python3
import argparse
import os
from typing import List
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
from pytope import Polytope
import rerun
import rerun.blueprint as rrb
import yaml
import torch
import rclpy
import uuid
from PIL import Image
import rclpy 
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from tools.utils import * 
import open_clip 
import time

from boxfusion.cubify_transformer import make_cubify_transformer

from boxfusion.instances import Instances3D
from boxfusion.preprocessor import Augmentor, Preprocessor

from boxfusion.box_manager import BoxManager
from boxfusion.box_fusion import BoxFusion
from tools.utils import points_in_box, mask_points_in_box

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from tools.utils import numpy_to_pc2
from tools.utils import get_objects_from_predicate_file
from my_msgs.msg import Box, BoxArray

def text_to_feature(strings, clip_model, device="cuda"):
    tokenized = open_clip.tokenize(strings).to(device)
    with torch.no_grad():
        feats = clip_model.encode_text(tokenized)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

# small ros node that publishes the online 3d detection boxes 
class Online3DNode(Node):
    def __init__(self):
        super().__init__('online_3d_node')

        # create publisher as list of PolygonStamped
        self.box_visual_publisher = self.create_publisher(MarkerArray, 'online_3d_boxes', 10)
        self.box_data_publisher = self.create_publisher(BoxArray, 'online_3d_box_data', 10)
        self.boxes = []

        # self.boxes = {'polygon_points': np.random.rand(8,3), 'label': 'None'}
        # self.publish_box(self.boxes)
        
        if not os.path.exists('./config/online.yaml'):
            raise ValueError("Missing config path")
        else:
            with open('./config/online.yaml', 'r') as  f:
                self.cfg = yaml.full_load(f)
        self.cfg['data']['output_dir'] = os.path.join(self.cfg['data']['output_dir'],self.cfg['recipe'])
        os.makedirs(self.cfg['data']['output_dir'], exist_ok=True)

        # extra objects, predicates from STL, to be detected!
        self.objects = get_objects_from_predicate_file(self.cfg['recipe'])
        print("Extra objects to be detected:", self.objects)
        
        # the kind of dataset (online)
        dataset = get_dataset(self.cfg)

        assert self.cfg['model_path'] is not None
        checkpoint = torch.load(self.cfg['model_path'], map_location=self.cfg['device'] or "cpu")["model"]
        backbone_embedding_dimension = checkpoint["backbone.0.patch_embed.proj.weight"].shape[0]
            
        is_depth_model = True
        model = make_cubify_transformer(dimension=backbone_embedding_dimension, depth_model=is_depth_model).eval()
        model.load_state_dict(checkpoint)

        dataset.load_arkit_depth = True
        augmentor = Augmentor(("wide/image", "wide/depth"))
        preprocessor = Preprocessor()
        
        if self.cfg['device'] is not None:
            model = model.to(self.cfg['device'])
            clip_model, preprocess = load_clip(self.cfg['clip_path'])
            text_class = np.genfromtxt(self.cfg['class_txt'], delimiter='\n', dtype=str) 
            text_class = np.char.lower(text_class)
            # print(f"text classes {text_class}")
            text_features = torch.load(self.cfg['class_features']).cuda()

            # self.objects = ['white cup', 'grey cup', 'black and red cup']
            # append extra_strings to text_class
            text_class = np.concatenate((text_class, np.array(self.objects)), axis=0)
            # append extra_strings features to text_features
            extra_features = text_to_feature(self.objects, clip_model, device=self.cfg['device'])
            text_features = torch.cat((text_features, extra_features), dim=0)
            # print(f"my embedding = their embedding?: {text_features[0,:] == text_to_feature([text_class[0]], clip_model, device=args.device)[0,:]}")
            # print("Loaded text features:", text_features.shape)
        
        self.run(model, dataset, clip_model, 
                 preprocess, text_class, text_features, augmentor, preprocessor, 
                 gap=25, re_vis=self.cfg['vis']['rerun']
        )
        self.save_boxes(save_path=self.cfg['data']['output_dir'])
        # obtain boxes that have not been detected
        not_detected = [obj for obj in self.objects if obj not in [box['label'] for box in self.boxes]]
        self.get_logger().warning(f"Not detected objects: {not_detected}")
        self.get_logger().warning(f"Either run again or define them yourself! (see manual_boxes.py)")

    def save_boxes(self,save_path:str=None):
        if save_path is None:
            save_path = self.cfg['data']['output_dir']
        """
        boxes: list of dicts
            [{'polygon_points': (8,3) ndarray, 'label': str}, ...]
        xyzrgb: (N,6) ndarray of the point cloud
        save_path: str
            path to save the boxes and point cloud

        Save all boxes as {'polygon_points': (8,3) ndarray, 
                           'cloud_points': (N,6) ndarray, 
                           'label': str} 
        """
        print(f"Saving {len(self.boxes)} boxes to: {save_path}")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i, box in enumerate(self.boxes):
            # make polytope from box corners2
            label = box['label']
            poly = Polytope(box['polygon_points'])
            in_box_points = points_in_box(poly, xyzrgb=self.xyzrgb)

            box_obj = {
                'polygon_points': box['polygon_points'],
                'cloud_points': in_box_points.cpu().numpy(),
                'label': label
            }

            with open(os.path.join(save_path, f"box_{i}_{label}.pkl"), "wb") as f:
                pickle.dump(box_obj, f)

    def publish_visual_boxes(self):
        """
        boxes: list of dicts
            [{'polygon_points': (8,3) ndarray, 'label': str}, ...]
        """
        print("Publishing visual boxes...")

        marker_array = MarkerArray()
        for i, box in enumerate(self.boxes):
            marker = Marker()
            marker.header.frame_id = "panda/panda_link0"
            marker.header.stamp = self.get_clock().now().to_msg()

            marker.ns = "online_3d_boxes"
            marker.id = i                     # IMPORTANT: unique per box
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD

            marker.scale.x = 0.01              # line width
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            corners = box['polygon_points']               # (8,3)

            edges = [
                (0,1),(1,2),(2,3),(3,0),
                (4,5),(5,6),(6,7),(7,4),
                (0,4),(1,5),(2,6),(3,7)
            ]

            for i0, i1 in edges:
                p0 = Point()
                p0.x = float(corners[i0][0])
                p0.y = float(corners[i0][1])
                p0.z = float(corners[i0][2])
                p1 = Point()
                p1.x = float(corners[i1][0])
                p1.y = float(corners[i1][1])
                p1.z = float(corners[i1][2])
                marker.points.append(p0)
                marker.points.append(p1)

            marker_array.markers.append(marker)

            self.box_visual_publisher.publish(marker_array)

    def publish_data_boxes(self):
        box_array_msg = BoxArray()
        # set header
        box_array_msg.header.frame_id = "panda/panda_link0"
        box_array_msg.header.stamp = self.get_clock().now().to_msg()

        for box in self.boxes:
            box_msg = Box()
            box_msg.header.stamp = self.get_clock().now().to_msg()
            box_msg.header.frame_id = "panda/panda_link0"

            # polygon (cheap)
            for x, y, z in box['polygon_points']:
                box_msg.polygon_points.append(Point(x=x, y=y, z=z))

            # cloud (fast)
            mask = mask_points_in_box(Polytope(box['polygon_points']), self.xyzrgb)
            box_msg.cloud_points = numpy_to_pc2(
                xyzrgb=self.xyzrgb[mask], 
                frame_id="panda/panda_link0", 
                stamp=self.get_clock().now().to_msg()
            )

            box_msg.label = box['label']
            box_array_msg.boxes.append(box_msg)
        for i, box in enumerate(self.boxes):
            box_msg = Box()
            # set header
            box_msg.header.frame_id = "panda/panda_link0"
            box_msg.header.stamp = self.get_clock().now().to_msg()

            # add box corners
            for corner in box['polygon_points']:
                point = Point()
                point.x = float(corner[0])
                point.y = float(corner[1])
                point.z = float(corner[2])
                box_msg.polygon_points.append(point)

            # add in-box points from the point cloud
            poly = Polytope(box['polygon_points'])
            in_box_points = points_in_box(poly, xyzrgb=self.xyzrgb)
            for pt in in_box_points:
                point = Point()
                point.x = float(pt[0])
                point.y = float(pt[1])
                point.z = float(pt[2])
                box_msg.cloud_points.append(point)

            # add label/label
            box_msg.label = box['label']
            box_array_msg.boxes.append(box_msg)

        self.box_data_publisher.publish(box_array_msg)

    def run(self, 
            model, dataset, clip_model, 
            preprocess, tokenized_text, text_features, augmentor, preprocessor, 
            gap=25, re_vis=True):
        
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

        box_manager = BoxManager(self.cfg)
        Box_Fuser = BoxFusion(self.cfg)

        box_count = 0
        start_time = time.time()
        print(f"STARTING TIMING")
        
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
            if self.cfg['viz_on_gt_points'] and sample["sensor_info"].has("gt"):
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

                pred_instances = pred_instances[pred_instances.scores >= float(self.cfg['detection']['score_thresh'])]
    
                # active filtering of unwanted boxes 
                if self.cfg["detection"]["uv_bound"]:
                    # check if the object is fully within the image
                    uv_mask = box_manager.check_uv_bounds(pred_instances.pred_proj_xy,
                                                          image.shape[1],image.shape[0],ratio=self.cfg["detection"]["uv_bound_value"]) #[N]
                    pred_instances = pred_instances[uv_mask]
                if self.cfg["detection"]["floor_mask"]:
                    # reject anisotropic boxes (too long in one dim compared to others)
                    floor_mask = box_manager.check_floor_mask(pred_instances.pred_boxes_3d.tensor, 
                                                              ratio=self.cfg["detection"]["floor_ratio"])
                    pred_instances = pred_instances[~floor_mask]

                # avoid first frame empty predictions
                if len(pred_instances) == 0 and count ==0:
                    with torch.no_grad():
                        pred_instances = model(packaged)[0]
                    pred_instances = pred_instances[pred_instances.scores >= float(self.cfg['detection']['score_thresh']/4)]
                    print("again",count,"pred_instances",len(pred_instances))
                    if self.cfg["detection"]["uv_bound"]:
                        uv_mask = box_manager.check_uv_bounds(pred_instances.pred_proj_xy,image.shape[1],image.shape[0],ratio=self.cfg["detection"]["uv_bound_value"]) #[N]
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
            if self.cfg["vis"]["trajectory"] and re_vis:
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
                    boxes = scale_boxes(boxes,image.shape[0],image.shape[1],scale=self.cfg['detection']['scale_box'])

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
                    mask, success_mask = Instances3D.spatial_association(all_pred_box,self.cfg["box_fusion"]["nms_threshold"],box_manager,per_frame_ins.cam_pose)
                    
                    cur_keep_idx = [i-num_before_cat for i in mask if i>=num_before_cat]
                    cur_success_nms = [i-num_before_cat for i in success_mask if i>=num_before_cat]
                    
    
                    keep_idx = np.asarray(mask)
                    if len(cur_keep_idx)>0:
                        '''
                        STEP2: correspondence association for small objects
                        '''
                        all_pred_box,all_poses,keep_idx = Instances3D.correspondence_association(
                            self.cfg, 
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
                            threshold=self.cfg['association']['small_threshold'],
                            H=image.shape[0],
                            W=image.shape[1]
                            )

                        # update the fusion list based on keep_idx
                        box_manager.update(keep_idx)
                    
                        print(count," box_manager",box_manager.fusion_list)

                        #filter those evident wrong boxes that valid_num=0
                        if self.cfg['box_fusion']['check_valid']:
                            all_pred_box = box_manager.check_valid_num(all_pred_box, count, gap)

                        '''
                        multi-view box fusion
                        '''
                        print("frame_id:box_num",box_manager.num_record)
                        if self.cfg['box_fusion']['use']:
                            Box_Fuser.boxfusion(all_pred_box, per_frame_ins, box_manager)
                    
                        #predict the semantic classes of remaining new boxes
                        cur_keep_idx = [i-num_before_cat for i in keep_idx if i>=num_before_cat]
                        cur_keep_idx_in_all = [i for i in range(keep_idx.shape[0]) if keep_idx[i]>=num_before_cat]

                        if len(cur_keep_idx)>0:
                            boxes = pred_instances.pred_boxes.cpu().numpy()
                            boxes = boxes[cur_keep_idx]
                            # scale the boxes
                            boxes = scale_boxes(boxes,image.shape[0],image.shape[1],scale=self.cfg['detection']['scale_box'])
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
                    visualize_online_boxes(all_pred_box, prefix="/device/wide", boxes_3d_name="pred_boxes_3d", log_instances_name="pred_instances",count=count,save=False,show_class=self.cfg["vis"]["show_class"],show_label=self.cfg["vis"]["show_label"]) 

            count+=1
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.cfg['total_duration']:
                print(f"Reached total duration of {self.cfg['total_duration']} seconds. Stopping capture.")
                break
            
        # save the results
        end_time = time.time()
        duration = end_time - start_time  
        fps = count / duration
        print(f"Cost: {duration:.2f} s", f"Average FPS: {fps:.2f}")
        
        # save global boxes for evaluation
        boxes_3d = all_pred_box.pred_boxes_3d.corners.cpu().numpy() # [N,8,3]
        print(f"centers after the loop: {all_pred_box.pred_boxes_3d.center}")
        if self.cfg["detection"]["workspace_filter"]:
            workspace_mask = box_manager.check_workspace_filter(
                all_pred_box.pred_boxes_3d.center,
                x_range=self.cfg["detection"]["workspace"]["x"],
                y_range=self.cfg["detection"]["workspace"]["y"],
                z_range=self.cfg["detection"]["workspace"]["z"]
            ).cpu()
            all_pred_box = all_pred_box[workspace_mask]
            boxes_3d = boxes_3d[workspace_mask]


        class_list = tokenized_text.tolist()
        class_idx = np.array([class_list.index(c) for c in all_pred_box.categories]) #[N]
        if self.cfg['data']['output_dir'] is not None:# and self.cfg["eval"]:
            if self.cfg['dataset'] == 'scannet':
                boxes_3d = post_process(boxes_3d)
                
            if boxes_3d.shape[0]>0:
                save_list = [[(int(n), (boxes_3d[n]), class_list[class_idx[n]]) for n in range(len(all_pred_box))]] # list of tuples class_idx[n]
                save_box(save_list, os.path.join(self.cfg['data']['output_dir'], "boxes.pkl"))

            # save the pointcloud, xyzrgb, (numpy object) to pkl
            print("Saving the pointcloud xyzrgb...")
            self.xyzrgb = xyzrgb
            torch.save(xyzrgb.cpu(), os.path.join(self.cfg['data']['output_dir'], 'xyzrgb.pt'))
        
        self.boxes = [{'polygon_points': boxes_3d[n], 'label': class_list[class_idx[n]]} for n in range(len(all_pred_box))]
        self.publish_visual_boxes()
            


def main(args=None):
    rclpy.init(args=args)
    node = Online3DNode()
    
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down online 3D node")
        node.destroy_node()
        rclpy.shutdown()



if __name__ == "__main__":
    main(args=None)
