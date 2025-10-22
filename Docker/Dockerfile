FROM osrf/ros:humble-desktop-full

WORKDIR /ros2_ws
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
# Remove any old/expired ROS 2 apt entry
RUN rm /etc/apt/sources.list.d/ros2-latest.list || true

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl gnupg2 lsb-release \
      iputils-ping \
      net-tools \
      build-essential gfortran libatlas-base-dev \
      python3 python3-pip python-is-python3 && \
    rm -rf /var/lib/apt/lists/*


# Re-import ROS 2 key and repo
#Comment this out if PGP key error occurs
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
    | gpg --dearmor \
    | tee /usr/share/keyrings/ros-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu \
      $(lsb_release -cs) main" \
    | tee /etc/apt/sources.list.d/ros2-latest.list

# Install Cyclone DDS
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ros-humble-rmw-cyclonedds-cpp && \
    rm -rf /var/lib/apt/lists/*

# Install RealSense + ROS 2 packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ros-humble-librealsense2* \
      ros-humble-realsense2-camera-msgs \
      ros-humble-tf2-ros\
      ros-humble-tf2-tools\
      ros-humble-realsense2-camera && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
      "numpy<2" \
      opencv-python \
      open3d \
      matplotlib \
      transformers \
      google-genai \
      Pillow \
      pyrealsense2

RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130


# Set CycloneDDS as RMW implementation
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
COPY ./src/h12_cameracalibration/Docker/CycloneConfigs/cyclonedds_robot.xml /etc/cyclonedds_robot.xml
COPY ./src/h12_cameracalibration/Docker/CycloneConfigs/cyclonedds_desktop.xml /etc/cyclonedds_desktop.xml
ENV CYCLONEDDS_URI=file:///etc/cyclonedds_desktop.xml


RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    ros-humble-ament-cmake \
    build-essential cmake git \
 && rm -rf /var/lib/apt/lists/*


WORKDIR /root
RUN git clone https://github.com/unitreerobotics/unitree_ros2.git
WORKDIR /root/unitree_ros2/cyclonedds_ws
RUN apt-get update && apt-get install -y ros-humble-rosidl-generator-dds-idl
RUN bash -c "source /opt/ros/humble/setup.sh && colcon build"
# WORKDIR /root
# RUN git clone https://github.com/correlllab/h12_ros2_controller.git
# WORKDIR /root/h12_ros2_controller
# RUN pip install -e .
# WORKDIR /root
# RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
# WORKDIR /root/unitree_sdk2_python
# RUN pip install -e .

WORKDIR /ros2_ws
# Entry point script to source and conditionally build
RUN echo '#!/usr/bin/env bash' > /ros_entrypoint.sh && \
    echo 'set -e' >> /ros_entrypoint.sh && \
    echo 'echo "Sourcing ROS 2..."' >> /ros_entrypoint.sh && \
    echo 'source /opt/ros/humble/setup.bash' >> /ros_entrypoint.sh && \
    echo '' >> /ros_entrypoint.sh && \
    echo 'if [ -f /root/unitree_ros2/cyclonedds_ws/install/setup.bash ]; then' >> /ros_entrypoint.sh && \
    echo '  echo "Sourcing Unitree workspace...";' >> /ros_entrypoint.sh && \
    echo '  source /root/unitree_ros2/cyclonedds_ws/install/setup.bash;' >> /ros_entrypoint.sh && \
    echo 'fi' >> /ros_entrypoint.sh && \
    echo '' >> /ros_entrypoint.sh && \
    echo 'if [ -d /ros2_ws/src/h12_cameracalibration ]; then' >> /ros_entrypoint.sh && \
    echo '  echo "Building h12_cameracalibration...";' >> /ros_entrypoint.sh && \
    echo '  colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release;' >> /ros_entrypoint.sh && \
    echo 'fi' >> /ros_entrypoint.sh && \
    echo '' >> /ros_entrypoint.sh && \
    echo 'if [ -f /ros2_ws/install/setup.bash ]; then' >> /ros_entrypoint.sh && \
    echo '  echo "Sourcing overlay workspace...";' >> /ros_entrypoint.sh && \
    echo '  source /ros2_ws/install/setup.bash;' >> /ros_entrypoint.sh && \
    echo 'fi' >> /ros_entrypoint.sh && \
    echo '' >> /ros_entrypoint.sh && \
    echo 'exec "$@"' >> /ros_entrypoint.sh && \
    chmod +x /ros_entrypoint.sh

RUN sudo apt update
RUN sudo apt install --only-upgrade ros-humble-geometry-msgs
RUN sudo apt install ros-humble-image-transport-plugins



ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
