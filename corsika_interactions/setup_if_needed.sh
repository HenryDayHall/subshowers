##### Setup corsika only if an install does not exist

echo "Received args"
echo $@

# Requires one argument - the installation directory

if [ "$#" -lt 1 ]; then
    echo "Usage: setup_if_needed.sh <install directory> [<env_name>]"
    exit 1
fi

# one optional argument, the name for the assocated python enviroment

if [ "$#" -gt 1 ]; then
    env_name=$2
else
    env_name="corsika"
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
test_script="$SCRIPT_DIR/test_install.sh"
install_script="$SCRIPT_DIR/install_no_fluka.sh"

$test_script "$1" "$env_name"
need_install=$?

if [ $need_install -eq 1 ]; then
    echo "Installing corsika"
    source "$install_script" "$1" "$env_name"

    echo "Running tests"
    $test_script "$1" "$env_name"
else
    echo "Corsika already installed"
fi

