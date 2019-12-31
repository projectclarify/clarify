
apt-get update

apt install g++ unzip zip wget -y

apt-get install openjdk-11-jdk -y

BAZEL_VERSION="2.0.0"
SCRIPT="bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh"

cd $HOME && wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/$SCRIPT

chmod +x $SCRIPT

./$SCRIPT --user

ln -s $HOME/bin/bazel /usr/local/bin/bazel