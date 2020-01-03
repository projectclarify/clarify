
BAZEL_VERSION="2.0.0"

SCRIPT="bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh"

cd $HOME && wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/$SCRIPT

chmod +x $SCRIPT

./$SCRIPT --user

echo "export PATH=$PATH:$HOME/bin" >> $HOME/.bashrc
