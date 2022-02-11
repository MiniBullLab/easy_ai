#!/usr/bin/env bash

# 错误码
ERR_MSG_DOCKER_NOT_INSTALL="Docker not installed."
ERR_MSG_DOCKER_SOCKET_PERMISSION="Docker socket permission deny."
ERR_MSG_NVIDIA_DOCKER_NOT_INSTALL="Nvidia docker not installed."
ERR_MSG_DOCKER_NOT_RUNNING="Docker not running."

EASY_PATH=/home/${USER}/easy_data

DOCKER_USER=minbull
AI_IMAGE=ai_runtime
AI_IMAGE_VERSION=1.0.1
WORKSPACE_IMAGE=ai_workspace

IMAGE_NAME=AI_IMAGE

# 运行环境检测失败，打印错误码并且退出
function envCheckFailedAndExit() {
   echo "EasyAI environment error."
   echo "Error msg: $1"
   exit 1
}

# 检测Docker是否安装
function checkDockerInstall() {
   docker --version | grep "Docker version" 1>/dev/null 2>&1
   # shellcheck disable=SC2181
   if [ $? != 0 ]; then
      envCheckFailedAndExit "$ERR_MSG_DOCKER_NOT_INSTALL"
   fi
}

function checkRuntimeImageExist() {
   docker image ls | grep "$AI_IMAGE_VERSION" | grep "$DOCKER_USER/$AI_IMAGE" 1>/dev/null 2>&1
   # shellcheck disable=SC2181
   if [ $? != 0 ]; then
      envCheckFailedAndExit "Image $DOCKER_USER/$AI_IMAGE:$AI_IMAGE_VERSION not exist!"
   fi
}

# 检测nvidia-docker是否安装
function checkNvidiaDocker() {
   nvidia-docker -v | grep 'Docker version' 1>/dev/null 2>&1
   # shellcheck disable=SC2181
   if [ $? != 0 ]; then
      envCheckFailedAndExit "$ERR_MSG_NVIDIA_DOCKER_NOT_INSTALL"
   fi
}

# 检验docker是否在运行
function checkDockerIsRunning() {
   checkResult=$(docker info --format '{{json .}}' | grep "Is the docker daemon running")
   if [ -n "$checkResult" ]; then
      envCheckFailedAndExit "$ERR_MSG_DOCKER_NOT_RUNNING"
   fi
}

# 检测docker socket权限
function checkDockerPermission() {
   checkResult=$(docker info --format '{{json .}}' | grep "Got permission denied while trying to connect to the Docker daemon socket")
   if [ -n "$checkResult" ]; then
      envCheckFailedAndExit "$ERR_MSG_DOCKER_SOCKET_PERMISSION"
   fi
}

# 检测运行环境
function checkRuntimeEnvironment() {
   echo "Begin check EasyAI environment..."
   checkDockerInstall
   checkDockerPermission
   checkDockerIsRunning
   checkNvidiaDocker
   echo "EasyAI environment OK"
   echo ""
}

# 启动加密狗
function startSenseshield() {
   echo "Begin start senseshield..."
   docker exec "$CONTAINER_NAME" sudo /usr/lib/senseshield/senseshield
   ps_result=$(docker exec "$CONTAINER_NAME" ps -aux | grep senseshield)
   if [ -z "$ps_result" ]; then
      echo "Start senseshield failed."
      docker stop "$CONTAINER_NAME" 1>/dev/null
      docker rm -v -f "$CONTAINER_NAME" 1>/dev/null
      exit 1
   fi
   echo "Start senseshield success."
   echo ""
}

# 在docker容器中创建对应的非root用户
function createDockerUser() {
   if [ "${USER}" != "root" ]; then
      echo ""
      echo "Current user is not root, begin to create docker user..."
      docker exec "${CONTAINER_NAME}" bash -c '/scripts/add_user.sh'
      # shellcheck disable=SC2181
      if [ $? == 0 ]; then
         echo "Create docker user success."
         echo ""
      else
         echo "Create docker user failed."
         exit 1
      fi
   fi
}

function main() {
   if [ "$1" = "workspace" ]; then
      IMAGE_NAME=$WORKSPACE_IMAGE
      FULL_IMAGE_NAME=$WORKSPACE_IMAGE
   else
      IMAGE_NAME=$AI_IMAGE
      FULL_IMAGE_NAME=$DOCKER_USER/$AI_IMAGE:$AI_IMAGE_VERSION
      checkRuntimeImageExist
   fi

   GRP_ID=$(id -g)
   GRP_NAME=$(id -g -n)
   USER_ID=$(id -u)
   USER_NAME=$(whoami)
   echo "Group id=$GRP_ID name=$GRP_NAME"
   echo "User id=$USER_ID name=$USER"
   echo ""

   checkRuntimeEnvironment

   if [ ! -d "$EASY_PATH" ]; then
      echo "easy_path not exist, create dir ${EASY_PATH}"
      mkdir "$EASY_PATH"
   fi

   CONTAINER_NAME="${IMAGE_NAME}_${USER_NAME}"
   docker ps -a --format "{{.Names}}" | grep "$CONTAINER_NAME" 1>/dev/null
   # shellcheck disable=SC2181
   if [ $? == 0 ]; then
      echo "${CONTAINER_NAME} is running, stop and remove..."
      docker stop "$CONTAINER_NAME" 1>/dev/null
      docker rm -v -f "$CONTAINER_NAME" 1>/dev/null
      echo "${CONTAINER_NAME} stop and remove success"
      echo ""
   fi

   echo "Starting docker container ${CONTAINER_NAME} ..."

   xhost +

   docker run -it --shm-size="2g" --gpus=all -d --privileged --name "$CONTAINER_NAME" \
      -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
      -e DOCKER_IMG="$IMAGE_NAME" \
      -e DOCKER_USER="$USER_NAME" \
      -e DOCKER_USER_ID="$USER_ID" \
      -e DOCKER_GRP="$GRP_NAME" \
      -e DOCKER_GRP_ID="$GRP_ID" \
      -v "${EASY_PATH}":/easy_data \
      "$FULL_IMAGE_NAME" \
      /bin/bash
   # shellcheck disable=SC2181
   if [ $? -ne 0 ]; then
      echo "Failed to start docker container \"${CONTAINER_NAME}\" based on image: $FULL_IMAGE_NAME"
      exit 1
   fi

   createDockerUser
   startSenseshield

   echo "Finished setting up EasyAi docker environment."
   echo "Now you can enter with: bash docker_into.sh"
}

main "$1"
