#!/bin/bash
##### Installing CORSIKA 8, with history retention, no FLUKA
# Installs the histories_1 branch of github.com/HenryDayHall/CORSIKA8
# without FLUKA, along with the conda environment it builds in.
#
# Usage: install_no_fluka.sh <install directory> [<env_name>]
#
# Inside <install directory> this creates
#   corsika/          the source
#   corsika-build/    the build tree, with make.log from the last compile
#   corsika-install/  the installed libraries and bin/
#
# Optional overrides, as environment variables:
#   CORSIKA_REPO        git URL to clone (default is https, so no SSH key needed)
#   CORSIKA_BRANCH      branch to build (default histories_1)
#   CORSIKA_BUILD_JOBS  parallel make jobs (default: all cores)
#
# Safe to rerun after a failure; an existing clone and build tree are reused.
# The clone gets one local edit: the Pythia 8 download address is updated,
# because pythia.org moved its downloads (see "Pythia 8 download address").

# setup_if_needed.sh sources this file. Rerun it as its own bash process so
# the `exit`, `cd` and `set -e` below can't leak into, or close, the caller.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    bash "${BASH_SOURCE[0]}" "$@"
    return
fi

set -euo pipefail

my_name=$(basename "$0")
trap 'echo "ERROR: $my_name stopped at line $LINENO: $BASH_COMMAND" >&2' ERR
die() { echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }

echo "$my_name received args: $*"

# Requires one argument - the installation directory
if [[ $# -lt 1 ]]; then
    echo "Usage: $my_name <install directory> [<env_name>]"
    exit 1
fi
# one optional argument, the name for the associated python environment
env_name=${2:-corsika}

python_version=3.14.3
repo=${CORSIKA_REPO:-https://github.com/HenryDayHall/CORSIKA8.git}
branch=${CORSIKA_BRANCH:-histories_1}
n_jobs=${CORSIKA_BUILD_JOBS:-$(nproc)}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
conda_yaml=$SCRIPT_DIR/corsika_conda.yaml
[[ -f "$conda_yaml" ]] || die "conda environment file not found: $conda_yaml"

# Absolute paths from here on, as we cd around below
mkdir -p -- "$1"
install_dir=$(CDPATH='' cd -- "$1" && pwd)
# CORSIKA's own build scripts don't quote paths, so spaces would break them
[[ "$install_dir" != *[[:space:]]* ]] || die "install directory can't contain spaces: $install_dir"
src_dir=$install_dir/corsika
build_dir=$install_dir/corsika-build
install_prefix=$install_dir/corsika-install


##### Conda environment
step "Setting up conda environment $env_name"
# `conda activate` needs conda's shell functions, which a non-interactive
# script doesn't get from your .bashrc, so load them here
conda_exe=${CONDA_EXE:-$(command -v conda || true)}
[[ -n "$conda_exe" ]] || die "conda not found, load or initialise conda first"
conda_hook=$("$conda_exe" shell.bash hook) || die "could not load conda's shell functions"
set +u  # conda's shell code isn't always safe with set -u
eval "$conda_hook"
set -u

# Match the name exactly; grep would also match e.g. corsika_old
if conda env list | awk -v name="$env_name" '$1 == name {found=1} END {exit !found}'; then
    echo "Using existing environment $env_name"
else
    echo "Creating new environment $env_name"
    conda create --yes -n "$env_name" "python=$python_version" \
        || die "failed to create environment $env_name"
fi

echo "Updating environment from $conda_yaml"
# -n is needed, otherwise conda updates the env named inside the yaml file.
# conda env update has no --yes flag (and doesn't prompt anyway).
conda env update -n "$env_name" --file "$conda_yaml" \
    || die "failed to update environment $env_name"

set +u
conda activate "$env_name" || die "failed to activate environment $env_name"
set -u

# check the python version
found_python=$(python -c 'import platform; print(platform.python_version())')
if [[ "$found_python" != "$python_version" ]]; then
    echo "WARNING: Python version is $found_python not $python_version"
    echo "This may cause issues, consider removing python env $env_name and rerunning this script"
fi

# Check for the build tools now, rather than an hour into the build
missing=()
for program in git cmake make g++ gfortran rsync tar conan; do
    command -v "$program" > /dev/null || missing+=("$program")
done
[[ ${#missing[@]} -eq 0 ]] || die "required programs not found: ${missing[*]}"

# conan crashes with "FileExistsError" if its home exists but isn't a folder.
# ~/.conan2 may be a symlink to a folder elsewhere (some tools ignore
# $CONAN_HOME), and deleting an old install can delete its target, so
# recreate the target if it's gone. The cache in it will be rebuilt.
conan_home=${CONAN_HOME:-$HOME/.conan2}
if [[ -L "$conan_home" && ! -e "$conan_home" ]]; then
    conan_target=$(readlink "$conan_home")
    # a relative link target is relative to the folder holding the link
    [[ "$conan_target" == /* ]] || conan_target=$(dirname "$conan_home")/$conan_target
    echo "Recreating $conan_target, which the symlink $conan_home points to"
    mkdir -p -- "$conan_target" || die "could not create $conan_target for $conan_home"
fi
if [[ -e "$conan_home" && ! -d "$conan_home" ]]; then
    die "conan's home $conan_home exists but isn't a folder; move it aside"
fi


##### Source
step "Getting CORSIKA source, branch $branch"
if [[ -d "$src_dir/.git" ]]; then
    echo "Reusing existing clone in $src_dir (delete it to start fresh)"
    git -C "$src_dir" submodule update --init --recursive \
        || die "failed to update submodules in $src_dir"
else
    # Clone into corsika/ explicitly; by default git would call it CORSIKA8/
    git clone --recursive --depth 1 -b "$branch" "$repo" "$src_dir" \
        || die "failed to clone $repo"
fi
echo "Building commit $(git -C "$src_dir" rev-parse --short HEAD)"


##### Pythia 8 download address
# pythia.org has moved its downloads: the address histories_1 fetches Pythia
# 8.315 from (.../download/pythia83/pythia8315.tar.bz2) now gives a 404, and
# the releases page only offers .tgz archives. Until the branch itself is
# fixed, point the build at the new address. The checksum is the one Spack and
# Nixpkgs both record for pythia8315.tgz, and CMake rejects a download that
# doesn't match it. Does nothing once the clone no longer has the old address.
pythia_cmake=$src_dir/modules/pythia8/CMakeLists.txt
old_pythia_dir=https://pythia.org/download/pythia83
new_pythia_dir=https://pythia.org/releases/pythia83
old_pythia_md5=faf2730a959369e4d25e1285ab70d915
new_pythia_sha256=4b2fe7341e33e90b7226fdcaa2a7bf9327987b3354e84c04f1fd9256863690ae
if [[ -f "$pythia_cmake" ]] && grep -qF "$old_pythia_dir" "$pythia_cmake"; then
    step "Pointing the Pythia 8 download at pythia.org's new address"
    # Only edit the file we expect: if the branch has moved to another Pythia
    # version, the checksum above would be wrong for it
    # shellcheck disable=SC2016  # ${_C8_Pythia8_VERSION} is CMake's, not ours
    for expected in '_C8_Pythia8_VERSION "8315"' \
                    'pythia${_C8_Pythia8_VERSION}.tar.bz2' \
                    "URL_MD5 $old_pythia_md5"; do
        grep -qF -- "$expected" "$pythia_cmake" \
            || die "can't update $pythia_cmake automatically, it doesn't contain: $expected"
    done
    # shellcheck disable=SC2016
    sed -i -e "s#$old_pythia_dir#$new_pythia_dir#" \
           -e 's#pythia${_C8_Pythia8_VERSION}\.tar\.bz2#pythia${_C8_Pythia8_VERSION}.tgz#' \
           -e "s#URL_MD5 $old_pythia_md5#URL_HASH SHA256=$new_pythia_sha256#" \
           "$pythia_cmake"
    if grep -qF -e "$old_pythia_dir" -e "$old_pythia_md5" "$pythia_cmake" \
        || ! grep -qF "URL_HASH SHA256=$new_pythia_sha256" "$pythia_cmake"; then
        die "updating $pythia_cmake didn't work, check it by hand"
    fi
    echo "Updated $pythia_cmake"
    echo "(a local change in the clone; commit it to $branch to make it permanent)"
fi


##### Build
mkdir -p "$build_dir"
cd "$build_dir"

step "Installing dependencies with conan (slow the first time)"
# This also generates corsika-cmake.sh in the source directory
bash "$src_dir/conan-install.sh" --source-directory "$src_dir" --release-with-debug \
    || die "conan-install.sh failed"
[[ -f "$src_dir/corsika-cmake.sh" ]] \
    || die "conan-install.sh did not create $src_dir/corsika-cmake.sh"

step "Configuring with cmake"
bash "$src_dir/corsika-cmake.sh" \
    -c "-DCMAKE_BUILD_TYPE=RelWithDebInfo -DWITH_FLUKA=OFF -DCMAKE_INSTALL_PREFIX=$install_prefix" \
    || die "cmake configuration failed"

step "Compiling with $n_jobs jobs"
make_log=$build_dir/make.log
if ! make -j"$n_jobs" 2>&1 | tee "$make_log"; then
    # In a parallel build the real error is printed long before make gives
    # up, and other jobs bury it, so pull the first errors back out
    echo
    echo "First errors in $make_log:"
    grep -n -i -m 10 -E "error:|CMake Error|\*\*\*" "$make_log" || true
    die "compilation failed, full output in $make_log"
fi
make install || die "make install failed"

echo
echo "Installed CORSIKA to $install_prefix"
