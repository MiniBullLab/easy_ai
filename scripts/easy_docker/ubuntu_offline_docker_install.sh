#!/bin/bash
#docker-ce
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common curl
curl -o containerd.io.deb http://118.31.19.101:8080/software/docker/containerd.io_1.4.3-1_amd64.deb
curl -o docker-ce-cli.deb http://118.31.19.101:8080/software/docker/docker-ce-cli_20.10.1_3-0_ubuntu-bionic_amd64.deb
curl -o docker-ce.deb http://118.31.19.101:8080/software/docker/docker-ce_20.10.1_3-0_ubuntu-bionic_amd64.deb
sudo dpkg -i containerd.io.deb
sudo dpkg -i docker-ce-cli.deb
sudo dpkg -i docker-ce.deb
sudo systemctl start docker
sudo systemctl enable docker

sudo docker version

rm -rf containerd.io.deb docker-ce-cli.deb docker-ce.deb

#nvidia-docker2
sudo apt-get purge -y nvidia-container-runtime nvidia-docker*
sudo apt -y autoremove

sudo pkill -SIGHUP dockerd
sudo systemctl restart docker
curl -o libnvidia-container1.deb http://118.31.19.101:8080/software/nvidia_docker/libnvidia-container1_1.3.1-1_amd64.deb
curl -o libnvidia-container-tools.deb http://118.31.19.101:8080/software/nvidia_docker/libnvidia-container-tools_1.3.1-1_amd64.deb
curl -o nvidia-container-toolkit.deb http://118.31.19.101:8080/software/nvidia_docker/nvidia-container-toolkit_1.4.0-1_amd64.deb
curl -o nvidia-container-runtime.deb http://118.31.19.101:8080/software/nvidia_docker/nvidia-container-runtime_3.4.0-1_amd64.deb
curl -o nvidia-docker2.deb http://118.31.19.101:8080/software/nvidia_docker/nvidia-docker2_2.5.0-1_all.deb
sudo dpkg -i libnvidia-container1.deb
sudo dpkg -i libnvidia-container-tools.deb
sudo dpkg -i nvidia-container-toolkit.deb
sudo dpkg -i nvidia-container-runtime.deb
sudo dpkg -i nvidia-docker2.deb
sudo pkill -SIGHUP dockerd
sudo systemctl restart docker

nvidia-docker -v

rm -rf libnvidia-container1.deb libnvidia-container-tools.deb nvidia-container-toolkit.deb nvidia-container-runtime.deb nvidia-docker2.deb

#nvidia-docker register
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/override.conf <<EOF
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd --host=fd:// --add-runtime=nvidia=/usr/bin/nvidia-container-runtime
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker

# Daemon configuration file
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo pkill -SIGHUP dockerd

# add user
sudo groupadd docker
sudo usermod -aG docker ${USER}
sudo systemctl restart docker
sudo chmod a+rw /var/run/docker.sock
sudo chgrp docker /var/run/docker.sock


