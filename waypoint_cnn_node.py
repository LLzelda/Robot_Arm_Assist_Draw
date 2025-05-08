#!/usr/bin/env python3
"""
ROS 2 node that reads the wrist‑camera RGB image, converts to grayscale,
runs the trained U‑Net, and publishes the predicted waypoint in board coords.
"""
import rclpy, cv2, torch, numpy as np, tf2_geometry_msgs
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
from torchvision.transforms import ToTensor

# ---------- same UNet -------
class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        def blk(ci, co):
            return torch.nn.Sequential(
                torch.nn.Conv2d(ci,co,3,1,1), torch.nn.BatchNorm2d(co), torch.nn.ReLU(),
                torch.nn.Conv2d(co,co,3,1,1), torch.nn.BatchNorm2d(co), torch.nn.ReLU())
        self.e1 = blk(1,32);  self.p1 = torch.nn.MaxPool2d(2)
        self.e2 = blk(32,64); self.p2 = torch.nn.MaxPool2d(2)
        self.b  = blk(64,128)
        self.u2 = torch.nn.ConvTranspose2d(128,64,2,2); self.d2 = blk(128,64)
        self.u1 = torch.nn.ConvTranspose2d(64,32,2,2);  self.d1 = blk(64,32)
        self.o  = torch.nn.Conv2d(32,1,1)
    def forward(self,x):
        e1=self.e1(x)
        e2=self.e2(self.p1(e1))
        b = self.b(self.p2(e2))
        d2=self.d2(torch.cat([self.u2(b),e2],1))
        d1=self.d1(torch.cat([self.u1(d2),e1],1))
        return self.o(d1)

# ---------- node -----------
class WaypointCNN(Node):
    def __init__(self):
        super().__init__('cnn_waypoint')
        self.declare_parameter('image', '/stretch/realsense/color/image_rect_raw')
        topic  = self.get_parameter('image').value
        self.br = CvBridge()
        self.pub = self.create_publisher(PointStamped, '/cnn_waypoint_board', 10)
        self.sub = self.create_subscription(Image, topic, self.cb_img, 10)

        self.tfbuf = Buffer(); self.tfl = TransformListener(self.tfbuf, self)
        self.net = UNet()
        self.net.load_state_dict(torch.load('unet_ep25.pth', map_location='cpu'))
        self.net.eval()
        self.to_tensor = ToTensor()
        self.get_logger().info(f'Listening to {topic}')

    def cb_img(self, msg):
        # decode & convert to grayscale
        rgb = self.br.imgmsg_to_cv2(msg, 'bgr8')
        gray= cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        inp = self.to_tensor(gray).unsqueeze(0)   # 1×1×H×W
        with torch.no_grad():
            heat = self.net(inp).sigmoid()[0,0].numpy()
        v,u = np.unravel_index(np.argmax(heat), heat.shape)  # row=v, col=u

        # build PointStamped in camera optical frame (z=1 as dummy)
        pt_cam = PointStamped()
        pt_cam.header = msg.header
        pt_cam.point.x, pt_cam.point.y, pt_cam.point.z = float(u), float(v), 1.0

        try:
            pt_board = self.tfbuf.transform(
                pt_cam, 'whiteboard',
                timeout=rclpy.duration.Duration(seconds=0.1))
            self.pub.publish(pt_board)
        except Exception as e:
            self.get_logger().warn(str(e))

# ---------- main -----------
def main():
    rclpy.init(); rclpy.spin(WaypointCNN()); rclpy.shutdown()

if __name__ == '__main__':
    main()
