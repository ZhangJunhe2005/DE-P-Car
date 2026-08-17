"""Uniformly scale a rendered URDF while preserving physical similarity."""

from xml.etree import ElementTree as ET


def _scaled_vector(value: str, factor: float) -> str:
    return " ".join(f"{float(component) * factor:.12g}" for component in value.split())


def _scaled_scalar(element: ET.Element, attribute: str, factor: float) -> None:
    if attribute in element.attrib:
        element.set(attribute, f"{float(element.get(attribute)) * factor:.12g}")


def configure_ros_depth_camera(xml_text: str) -> str:
    """Replace the unavailable RealSense plugin with gazebo_ros depth camera.

    The pinned Urban model describes a D435 using four Gazebo sensors and an
    external ``librealsense_gazebo_plugin``.  Ubuntu/ROS Noetic already ships
    ``libgazebo_ros_depth_camera``, so retain the D435 links and depth sensor,
    remove the unused color/IR sensors, and publish a compact 640x480 stream at
    15 Hz under ``/camera``.
    """

    root = ET.fromstring(xml_text)
    found_realsense = False
    for gazebo in list(root.findall("gazebo")):
        for plugin in list(gazebo.findall("plugin")):
            if plugin.get("filename") == "librealsense_gazebo_plugin.so":
                gazebo.remove(plugin)
                found_realsense = True
        if not list(gazebo) and not gazebo.attrib:
            root.remove(gazebo)

    # With enable_camera:=false there is nothing to replace.
    if not found_realsense:
        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    camera_gazebo = next(
        (item for item in root.findall("gazebo") if item.get("reference") == "camera_link"),
        None,
    )
    if camera_gazebo is None:
        raise ValueError("D435 camera_link Gazebo element is missing")

    depth_sensor = None
    for sensor in list(camera_gazebo.findall("sensor")):
        if sensor.get("type") == "depth":
            depth_sensor = sensor
        else:
            camera_gazebo.remove(sensor)
    if depth_sensor is None:
        raise ValueError("D435 depth sensor is missing")

    update_rate = depth_sensor.find("update_rate")
    if update_rate is None:
        update_rate = ET.SubElement(depth_sensor, "update_rate")
    update_rate.text = "15.0"
    camera = depth_sensor.find("camera")
    image = camera.find("image") if camera is not None else None
    if image is None:
        raise ValueError("D435 depth image description is missing")
    width = image.find("width")
    height = image.find("height")
    if width is None or height is None:
        raise ValueError("D435 depth image dimensions are missing")
    width.text, height.text = "640", "480"
    clip = camera.find("clip")
    if clip is None:
        clip = ET.SubElement(camera, "clip")
    near = clip.find("near")
    far = clip.find("far")
    if near is None:
        near = ET.SubElement(clip, "near")
    if far is None:
        far = ET.SubElement(clip, "far")
    near.text, far.text = "0.2", "10.0"

    plugin = ET.SubElement(
        depth_sensor,
        "plugin",
        {"name": "dep_car_depth_camera_controller", "filename": "libgazebo_ros_depth_camera.so"},
    )
    values = {
        "alwaysOn": "true",
        "updateRate": "0.0",
        "cameraName": "camera",
        "imageTopicName": "color/image_raw",
        "cameraInfoTopicName": "color/camera_info",
        "depthImageTopicName": "depth/image_raw",
        "depthImageCameraInfoTopicName": "depth/camera_info",
        "pointCloudTopicName": "depth/color/points",
        "frameName": "camera_depth_optical_frame",
        "pointCloudCutoff": "0.2",
        "pointCloudCutoffMax": "10.0",
    }
    for name, value in values.items():
        ET.SubElement(plugin, name).text = value

    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def configure_sim_lidar(
    xml_text: str,
    ray_min=0.15,
    valid_min=0.20,
    valid_max=40.0,
    mount_xyz=(0.0, 0.0, 1.75),
) -> str:
    """Apply the versioned close-range simulation contract to VLP-16.

    The physical VLP-16 default minimum range (0.9 m) leaves a disproportionate
    blind ring around the one-third-scale Urban Car.  The training simulator
    deliberately uses a 0.2 m valid minimum; this difference is recorded in
    the sensor contract and must not be assumed for real-hardware transfer.
    """

    root = ET.fromstring(xml_text)
    configured = False
    mount_joint = root.find(".//joint[@name='velodyne_base_mount_joint']")
    if mount_joint is not None:
        origin = mount_joint.find("origin")
        if origin is None:
            origin = ET.SubElement(mount_joint, "origin")
        # This value is applied before uniform scaling.  At one-third scale the
        # base is z=0.583 m and the scan plane is about z=0.596 m, above the
        # 0.517 m body so the promised 360-degree field is not self-occluded.
        origin.set("xyz", " ".join(str(float(value)) for value in mount_xyz))
    for sensor in root.findall(".//sensor"):
        plugins = sensor.findall("plugin")
        velodyne_plugins = [
            plugin for plugin in plugins if "velodyne" in (plugin.get("filename") or "").lower()
        ]
        if not velodyne_plugins:
            continue
        range_element = sensor.find("ray/range")
        if range_element is None:
            raise ValueError("VLP-16 ray range description is missing")
        minimum = range_element.find("min")
        maximum = range_element.find("max")
        if minimum is None or maximum is None:
            raise ValueError("VLP-16 minimum/maximum ray range is missing")
        minimum.text = str(float(ray_min))
        maximum.text = str(float(valid_max) + 1.0)
        for plugin in velodyne_plugins:
            plugin_minimum = plugin.find("min_range")
            plugin_maximum = plugin.find("max_range")
            if plugin_minimum is None or plugin_maximum is None:
                raise ValueError("VLP-16 plugin range contract is missing")
            plugin_minimum.text = str(float(valid_min))
            plugin_maximum.text = str(float(valid_max))
        configured = True
    if not configured:
        return xml_text
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def scale_urdf_xml(xml_text: str, linear_scale: float) -> str:
    """Return a uniformly scaled URDF.

    Lengths use ``s``, masses use ``s^3``, inertias use ``s^5``, and joint
    efforts/damping use ``s^4``. Sensor measurement ranges are deliberately
    left unchanged because the simulated VLP-16 still measures the same world.
    """

    scale = float(linear_scale)
    if not 0.0 < scale <= 1.0:
        raise ValueError("linear_scale must be in the interval (0, 1]")

    root = ET.fromstring(xml_text)
    root.set("name", f"{root.get('name', 'urban')}_scaled")

    for origin in root.findall(".//origin"):
        if origin.get("xyz"):
            origin.set("xyz", _scaled_vector(origin.get("xyz"), scale))

    for mesh in root.findall(".//mesh"):
        mesh.set("scale", _scaled_vector(mesh.get("scale", "1 1 1"), scale))
    for box in root.findall(".//box"):
        if box.get("size"):
            box.set("size", _scaled_vector(box.get("size"), scale))
    for cylinder in root.findall(".//cylinder"):
        _scaled_scalar(cylinder, "radius", scale)
        _scaled_scalar(cylinder, "length", scale)
    for sphere in root.findall(".//sphere"):
        _scaled_scalar(sphere, "radius", scale)

    for mass in root.findall(".//mass"):
        _scaled_scalar(mass, "value", scale ** 3)
    for inertia in root.findall(".//inertia"):
        for attribute in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
            _scaled_scalar(inertia, attribute, scale ** 5)

    for limit in root.findall(".//limit"):
        _scaled_scalar(limit, "effort", scale ** 4)
    for dynamics in root.findall(".//dynamics"):
        _scaled_scalar(dynamics, "damping", scale ** 4)
        _scaled_scalar(dynamics, "friction", scale ** 4)
    for minimum_depth in root.findall(".//minDepth"):
        if minimum_depth.text:
            minimum_depth.text = f"{float(minimum_depth.text) * scale:.12g}"

    return ET.tostring(root, encoding="unicode", xml_declaration=True)
