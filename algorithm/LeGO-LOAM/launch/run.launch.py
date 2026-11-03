import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():

  # Configure environment
  stdout_linebuf_envvar = SetEnvironmentVariable('RCUTILS_CONSOLE_STDOUT_LINE_BUFFERED', '1')
  stdout_colorized_envvar = SetEnvironmentVariable('RCUTILS_COLORIZED_OUTPUT', '0')

  # Simulated time
  use_sim_time = LaunchConfiguration('use_sim_time', default='true')
  use_neural_ground = LaunchConfiguration('use_neural_ground')
  diagnostics = LaunchConfiguration('diagnostics')
  rviz = LaunchConfiguration('rviz')
  onnx_path = LaunchConfiguration('onnx_path')
  enable_end_to_end_timing = LaunchConfiguration('enable_end_to_end_timing')

  declare_use_neural_ground = DeclareLaunchArgument('use_neural_ground', default_value='false')
  declare_diagnostics = DeclareLaunchArgument('diagnostics', default_value='false')
  declare_rviz = DeclareLaunchArgument('rviz', default_value='true')
  package_share = get_package_share_directory('lego_loam_sr')
  
  default_onnx_path = os.path.join(
    package_share,
    'model',
    'ground_segmentation.onnx'
  )
  declare_onnx_path = DeclareLaunchArgument(
    'onnx_path',
    default_value=default_onnx_path,
    description='Path to the ONNX ground segmentation model'
  )
  declare_enable_end_to_end_timing = DeclareLaunchArgument(
    'enable_end_to_end_timing',
    default_value='false',
    description='Enable end-to-end latency measurements'
  )

  # Nodes Configurations
  config_file = os.path.join(get_package_share_directory('lego_loam_sr'), 'config', 'loam_config.yaml')
  rviz_config = os.path.join(get_package_share_directory('lego_loam_sr'), 'rviz', 'test.rviz')

  # Tf transformations
  transform_map = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    name='camera_init_to_map',
    arguments=['0', '0', '0', '1.570795', '0', '1.570795', 'map', 'camera_init'],
  )

  transform_camera = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    name='base_link_to_camera',
    arguments=['0', '0', '0', '-1.570795', '-1.570795', '0', 'camera', 'base_link'],
  )

  transform_velodyne = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    name='velodyne_to_base_link',
    arguments=['0', '0', '0', '0', '0', '0','base_link','velodyne'],
  )

  # LeGO-LOAM
  lego_loam_node = Node(
    package='lego_loam_sr',
    executable='lego_loam_sr',
    output='screen',
    parameters=[config_file, {
      'image_projection.use_neural_ground': use_neural_ground,
      'image_projection.diagnostics': diagnostics,
      'image_projection.onnx_path': onnx_path,
      'image_projection.enable_end_to_end_timing': enable_end_to_end_timing,
    }],
    remappings=[('/rslidar_points', '/velodyne_points')],
  )

  # Rviz
  rviz_node = Node(
    condition=IfCondition(rviz),
    package='rviz2',
    executable='rviz2',
    name='rviz2',
    arguments=['-d', rviz_config],
    output='screen'
  )

  ld = LaunchDescription()

  # Set environment variables
  ld.add_action(stdout_linebuf_envvar)
  ld.add_action(stdout_colorized_envvar)

  # Declare launch arguments
  ld.add_action(declare_use_neural_ground)
  ld.add_action(declare_diagnostics)
  ld.add_action(declare_rviz)
  ld.add_action(declare_onnx_path)
  ld.add_action(declare_enable_end_to_end_timing)

  # Add nodes
  ld.add_action(lego_loam_node)
  ld.add_action(transform_map)
  ld.add_action(transform_camera)
  ld.add_action(transform_velodyne)
  ld.add_action(rviz_node)

  return ld