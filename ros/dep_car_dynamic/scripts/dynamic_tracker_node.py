#!/usr/bin/env python3
"""Track perception observations. Simulator/Pedsim ground truth is forbidden."""

import rospy
from dep_car.dynamic.tracker import ConstantVelocityTracker
from dep_car_msgs.msg import DynamicTrack, DynamicTrackArray
from geometry_msgs.msg import PoseArray


class DynamicTrackerNode:
    def __init__(self):
        self.tracker = ConstantVelocityTracker()
        self.publisher = rospy.Publisher("/dep_car/dynamic/tracks", DynamicTrackArray, queue_size=1)
        rospy.Subscriber("/dep_car/dynamic/observations", PoseArray, self.callback, queue_size=1)

    def callback(self, message):
        observations = [(pose.position.x, pose.position.y) for pose in message.poses]
        tracks = self.tracker.update(observations, message.header.stamp.to_sec())
        output = DynamicTrackArray()
        output.header = message.header
        for item in tracks:
            track = DynamicTrack()
            track.header = message.header
            track.track_id = item.track_id
            track.x, track.y, track.vx, track.vy = item.x, item.y, item.vx, item.vy
            track.radius = item.radius
            track.confidence = item.confidence
            track.position_covariance = item.covariance[:2, :2].flatten().tolist()
            output.tracks.append(track)
        self.publisher.publish(output)


if __name__ == "__main__":
    rospy.init_node("dep_car_dynamic_tracker")
    DynamicTrackerNode()
    rospy.spin()

