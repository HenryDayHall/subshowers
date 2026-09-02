##### Installing Corsika 8, with history retention, no fluka
# This script should install my branch of corsika 8
# sans fluka.
my_name=install_no_fluka.sh
echo "$my_name received args"
echo $@

# Requires one argument - the installation directory

if [ "$#" -lt 1 ]; then
    echo "Usage: $my_name <install directory> [<env_name>]"
    exit 1
fi

# one optional argument, the name for the assocated python enviroment

if [ "$#" -gt 1 ]; then
    env_name=$2
else
    env_name="corsika"
fi

# check if the env already exists

if conda env list | grep -q "$env_name"; then
    echo "Using existing environment $env_name"
else
    echo "Creating new environment $env_name"
    conda create --yes -n $env_name python=3.14.3
fi

conda activate $env_name
# check the python version
if [ "$(python --version)" != "Python 3.14.3" ]; then
    echo "Python version is $(python --version) not 3.14.3"
    echo "This may cause issues, consider removing python env $env_name and rerunning this script"
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
conda_yaml=$SCRIPT_DIR/corsika_conda.yaml

echo "Updating environment from $conda_yaml"
conda env update --yes --file $conda_yaml


install_dir=$1
mkdir -p $install_dir

# my repo is github.com/HenryDayHall/CORSIKA8
# and the branch is histories_1
cd $install_dir

git clone --recursive -b histories_1 --depth 1 git@github.com:HenryDayHall/CORSIKA8.git
# check the clone was successful
ret_code=$?
if [ $ret_code -ne 0 ]; then
    echo "Failed to clone CORSIKA8"
    exit $ret_code
fi

mkdir corsika-build
cd corsika-build
../corsika/corsika-cmake.sh -c "-DCMAKE_BUILD_TYPE="RelWithDebInfo" -DWITH_FLUKA=OFF -DCMAKE_INSTALL_PREFIX=../corsika-install"
make -j$(nproc)
make install

echo "Installed CORSIKA to $install_dir/corsika-install"


