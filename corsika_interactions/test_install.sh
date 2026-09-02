##### Test an installation of Corsika 8, returning 1 if it fails
my_name=test_install.sh
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

# check the conda env exists

if conda env list | grep -q "$env_name"; then
    echo "Conda environment $env_name exists"
else
    echo "Conda environment $env_name does not exist"
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$env_name"

# check the python version
if [ "$(python --version)" != "Python 3.14.3" ]; then
    echo "Python version is $(python --version) not 3.14.3"
    echo "This may cause issues, consider removing python env $env_name and rerunning this script"
fi

# check the packages in the env include those in the conda yaml file
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
conda_yaml=$SCRIPT_DIR/corsika_conda.yaml

# conda env update has no --dry-run in this conda version, so compare the
# yaml's dependency list against what's actually installed in the env instead
required_pkgs=$(awk '/^dependencies:/{flag=1; next} /^[^[:space:]]/{flag=0} flag' "$conda_yaml" \
    | sed -E 's/^[[:space:]]*-[[:space:]]*//; s/[<>=!].*//')
installed_pkgs=$(conda list --name "$env_name" | grep -v '^#' | awk '{print $1}')

missing_pkgs=()
for pkg in $required_pkgs; do
    if ! grep -qxi "$pkg" <<< "$installed_pkgs"; then
        missing_pkgs+=("$pkg")
    fi
done

if [ ${#missing_pkgs[@]} -eq 0 ]; then
    echo "Required packages installed in $env_name"
else
    echo "Enviroment $env_name does not contain required packages"
    echo "Missing packages:"
    printf '  %s\n' "${missing_pkgs[@]}"
    exit 1
fi

# check the install dir exists
install_dir=$1
if [ ! -d "$install_dir" ]; then
    echo "Install directory $install_dir does not exist"
    exit 1
fi

# check the build dir exists

if [ ! -d "$install_dir/corsika-build" ]; then
    echo "Build directory $install_dir/corsika-build does not exist"
    exit 1
fi

# check the applications dir exists

if [ ! -d "$install_dir/corsika-build/applications" ]; then
    echo "Applications directory $install_dir/corsika-build/applications does not exist"
    exit 1
fi

# check that c8_air_shower_with_history is in the applications dir
if [ ! -f "$install_dir/corsika-build/applications/c8_air_shower_with_history" ]; then
    echo "c8_air_shower_with_history does not exist in $install_dir/corsika-build/applications"
    exit 1
fi

#### not currently unit testing because they don't work
# cd $install_dir/corsika-build
# # run the tests, printing as we go and save the output
# test_outcome=$(ctest -j$(nproc) 2>&1 | tee /dev/tty)
# 
# if echo "$test_outcome" | grep -q "Failed: 0"; then
#     echo "Tests passed - all finished"
# else
#     echo "Tests failed - consider reinstall"
#     exit 1
# fi


