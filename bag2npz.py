"""
Convert a ROS 2 bag to (image, heatmap) npz pairs.
  • Uses /pen_tip_board for current tip
  • Looks 1 cm ahead along velocity for waypoint
  • Projects boardcoords → camera pixels with TF
"""
import os, cv2, numpy as np, argparse, rclpy, tf2_ros, tf2_geometry_msgs
from rclpy.time import Time
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from geometry_msgs.msg import PointStamped

def gaussian(shape, center, sigma=5):
    y,x = np.ogrid[:shape[0], :shape[1]]
    return np.exp(-((x-center[0])**2 + (y-center[1])**2)/(2*sigma**2))

# ---------------- CLI ----------------
ap = argparse.ArgumentParser()
ap.add_argument('--bag', required=True)
ap.add_argument('--dst', required=True)
ap.add_argument('--forward_cm', type=float, default=1.0)
args = ap.parse_args(); os.makedirs(args.dst, exist_ok=True)

# ---------------- set up ----------------
rclpy.init()
buf = tf2_ros.Buffer(); tf2_ros.TransformListener(buf, rclpy.node.Node('tmp'))
reader = SequentialReader()
reader.open(StorageOptions(uri=args.bag, storage_id='sqlite3'),
            ConverterOptions('', ''))
topics = reader.get_all_topics_and_types()
img_topic  = '/stretch/realsense/color/image_rect_raw'
tip_topic  = 'pen_tracker' #change this if I rembmer wrong
cache_tip  = {}

step = 0
while reader.has_next():
    topic, data, t = reader.read_next()
    if topic == tip_topic:
        cache_tip[t] = PointStamped.deserialize(data).point
        continue
    if topic != img_topic:
        continue

    img_msg = cv2.imdecode(np.frombuffer(data.data, dtype=np.uint8),
                           cv2.IMREAD_COLOR)
    if img_msg is None: continue
    stamp  = Time(seconds=data.header.stamp.sec,
                  nanoseconds=data.header.stamp.nanosec).to_msg()

    #need tip pose at this stamp and a bit ahead
    tip_now = cache_tip.get(data.header.stamp.nanosec)
    tip_fwd = cache_tip.get(data.header.stamp.nanosec + int(args.forward_cm*1e7))
    if tip_now is None or tip_fwd is None:
        continue

    #convert board meters → camera pixels (simple homography H you calibrate)
    def board2pix(pt):
        X,Y = pt.x, pt.y
        u = int(H[0,0]*X + H[0,1]*Y + H[0,2]) / (H[2,0]*X + H[2,1]*Y + 1)
        v = int(H[1,0]*X + H[1,1]*Y + H[1,2]) / (H[2,0]*X + H[2,1]*Y + 1)
        return u,v
    u,v = board2pix(tip_fwd)

    heat = gaussian(img_msg.shape[:2], (u,v), sigma=5).astype(np.float32)
    out  = os.path.join(args.dst, f'{step:05d}.npz')
    np.savez_compressed(out, img=img_msg, hm=heat)
    step += 1
    if step % 500 == 0:
        print('wrote', step)

print('Done:', step, 'pairs saved')
