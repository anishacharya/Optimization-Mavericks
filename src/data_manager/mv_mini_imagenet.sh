curr_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA_ROOT=$curr_dir'/data'

mkdir -p "$DATA_ROOT"/mini-imagenet
# shellcheck disable=SC2164
cd "$DATA_ROOT"/mini-imagenet
cp ~/mini-imagenet.tar.gz .
tar -xzvf mini-imagenet.tar.gz
rm -f mini-imagenet.tar.gz