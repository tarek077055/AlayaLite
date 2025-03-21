
cat << 'EOF' > Dockerfile.pypi
FROM quay.io/pypa/manylinux_2_28_x86_64
WORKDIR /source  
COPY . .
RUN yum install wget -y
RUN wget -q -O /tmp/conda.sh https://mirrors.sustech.edu.cn/anaconda/miniconda/Miniconda3-py311_25.1.1-1-Linux-x86_64.sh && \
      sh /tmp/conda.sh -b -p /opt/conda && \
      eval "$(/opt/conda/bin/conda shell.bash hook)" && \
      conda init &&\
      conda config --add channels https://mirrors.sustech.edu.cn/anaconda/pkgs/free/ &&\
      conda config --add channels https://mirrors.sustech.edu.cn/anaconda/pkgs/main/ &&\
      pip install --upgrade pip --index-url https://mirrors.sustech.edu.cn/pypi/web/simple &&\
      pip config set global.index-url https://mirrors.sustech.edu.cn/pypi/web/simple &&\
      conda create -n alayalite gcc=13 gxx=13  python=3.11 build -c conda-forge --yes &&\
      conda activate alayalite &&\
      python -m build --wheel && \
      auditwheel repair "dist/$(ls dist | head -n 1)" --wheel-dir dist/
EOF
echo "Dockerfile.pypi created successfully"

rm -rf dist
if [ "$(docker images -q alayalite/build_pypi 2> /dev/null)" != "" ]; then
  echo "Docker image alayalite/build_pypi exists, removing it"
  docker rmi alayalite/build_pypi
fi

echo "RUN: docker build -f Dockerfile.pypi -t alayalite/build_pypi ."
docker build -f Dockerfile.pypi -t alayalite/build_pypi .
echo "Docker image alayalite/build_pypi created successfully"

echo "cp whl files to dist"
docker create --name temp_alayalite_build_pypi alayalite/build_pypi
docker cp temp_alayalite_build_pypi:/source/dist $(pwd)/dist
docker rm temp_alayalite_build_pypi
echo "cp whl files to dist successfully"
rm Dockerfile.pypi
echo "Dockerfile.pypi removed successfully"
echo "Build pypi wheel successfully"