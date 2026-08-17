import xml.etree.ElementTree as ET

import pytest

from dep_car.core.occupancy import FootprintConfig
from dep_car.core.urdf_scale import configure_ros_depth_camera, configure_sim_lidar, scale_urdf_xml
from dep_car.core.vehicle import (
    LENGTH_M,
    URBAN_CAR_LINEAR_SCALE,
    WHEELBASE_M,
    WHEEL_RADIUS_M,
    WIDTH_M,
)


def test_authoritative_geometry_is_one_third_of_upstream():
    assert URBAN_CAR_LINEAR_SCALE == pytest.approx(1.0 / 3.0)
    assert WHEELBASE_M == pytest.approx(0.98 / 3.0)
    assert LENGTH_M == pytest.approx(1.49 / 3.0)
    assert WIDTH_M == pytest.approx(0.83 / 3.0)
    assert WHEEL_RADIUS_M == pytest.approx(0.19 / 3.0)
    assert FootprintConfig().length == pytest.approx(LENGTH_M)
    assert FootprintConfig().width == pytest.approx(WIDTH_M)


def test_urdf_scaler_applies_similarity_laws_but_keeps_sensor_range():
    source = """<robot name="urban">
      <link name="body">
        <visual><origin xyz="3 0 0"/><geometry><mesh filename="body.stl" scale="1 1 1"/></geometry></visual>
        <collision><geometry><box size="3 6 9"/></geometry></collision>
        <inertial><mass value="27"/><inertia ixx="243" ixy="0" ixz="0" iyy="243" iyz="0" izz="243"/></inertial>
      </link>
      <joint name="wheel" type="continuous"><limit effort="81" velocity="10"/><dynamics damping="81" friction="81"/></joint>
      <gazebo><minDepth>0.03</minDepth><sensor><ray><range><max>40</max></range></ray></sensor></gazebo>
    </robot>"""
    root = ET.fromstring(scale_urdf_xml(source, 1.0 / 3.0))
    assert root.get("name") == "urban_scaled"
    assert root.find(".//origin").get("xyz") == "1 0 0"
    assert root.find(".//mesh").get("scale") == "0.333333333333 0.333333333333 0.333333333333"
    assert root.find(".//box").get("size") == "1 2 3"
    assert float(root.find(".//mass").get("value")) == pytest.approx(1.0)
    assert float(root.find(".//inertia").get("ixx")) == pytest.approx(1.0)
    assert float(root.find(".//limit").get("effort")) == pytest.approx(1.0)
    assert root.find(".//limit").get("velocity") == "10"
    assert float(root.find(".//minDepth").text) == pytest.approx(0.01)
    assert root.find(".//max").text == "40"


def test_realsense_plugin_is_replaced_by_installed_depth_camera_plugin():
    source = """<robot name="urban">
      <link name="camera_depth_optical_frame"/>
      <gazebo reference="camera_link">
        <sensor name="cameracolor" type="camera"><update_rate>30</update_rate></sensor>
        <sensor name="cameradepth" type="depth"><camera><image><width>1280</width><height>720</height></image></camera><update_rate>90</update_rate></sensor>
      </gazebo>
      <gazebo><plugin name="camera" filename="librealsense_gazebo_plugin.so"/></gazebo>
    </robot>"""
    root = ET.fromstring(configure_ros_depth_camera(source))
    plugins = root.findall(".//plugin")
    assert [item.get("filename") for item in plugins] == ["libgazebo_ros_depth_camera.so"]
    assert len(root.findall(".//sensor")) == 1
    sensor = root.find(".//sensor")
    assert sensor.get("type") == "depth"
    assert sensor.find("update_rate").text == "15.0"
    assert sensor.find("camera/image/width").text == "640"
    assert sensor.find("camera/image/height").text == "480"
    assert sensor.find("plugin/depthImageTopicName").text == "depth/image_raw"
    assert sensor.find("plugin/frameName").text == "camera_depth_optical_frame"
    assert sensor.find("camera/clip/near").text == "0.2"
    assert sensor.find("camera/clip/far").text == "10.0"


def test_sim_lidar_close_range_contract_is_applied_before_scaling():
    source = """<robot name="urban"><joint name="velodyne_base_mount_joint" type="fixed"><origin xyz="0.3 0 1.3"/></joint><gazebo reference="velodyne"><sensor type="ray"><ray><range><min>0.3</min><max>131</max></range></ray><plugin filename="libgazebo_ros_velodyne_laser.so"><min_range>0.9</min_range><max_range>130</max_range></plugin></sensor></gazebo></robot>"""
    root = ET.fromstring(configure_sim_lidar(source))
    assert root.find(".//ray/range/min").text == "0.15"
    assert root.find(".//ray/range/max").text == "41.0"
    assert root.find(".//plugin/min_range").text == "0.2"
    assert root.find(".//plugin/max_range").text == "40.0"
    assert [float(value) for value in root.find(".//joint/origin").get("xyz").split()] == pytest.approx([0.0, 0.0, 1.75])
    scaled = ET.fromstring(scale_urdf_xml(ET.tostring(root, encoding="unicode"), 1.0 / 3.0))
    assert [float(value) for value in scaled.find(".//joint/origin").get("xyz").split()] == pytest.approx([0.0, 0.0, 1.75 / 3.0])
