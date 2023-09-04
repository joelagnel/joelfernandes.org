#!/bin/bash
# This script should be copied from my snips/rcfiles/sh
# Do not modify

if [ ! -z $1 ] && [ $1 == "--debug" ]; then
	set -x
fi
set -e

# For background and instructions, see /j/rcfiles/ssh-files/README

spath="$(dirname "$(readlink -f "$0")")"

echo "Setting up machine"

sudo apt-get install curl git p7zip-full

rm -rf $spath/tmp-ssh-files/
mkdir $spath/tmp-ssh-files/
curl -sL http://joelfernandes.org/ssh-files/dot_ssh.7z -o $spath/tmp-ssh-files/dot_ssh.7z

echo -n Password:
read -s password

mkdir -p $HOME/.ssh
# TODO: On devices with root, like a chroot, this causes weird UID/GID in
# /root/.ssh and causes git clone to fail, find a better way?
pushd $HOME
rm -rf $HOME/.ssh.bak/
mv $HOME/.ssh/ $HOME/.ssh.bak || true
# Was compressed using: 7z -p$pass a $spath/tmp-ssh-files/dot_ssh.7z ~/.ssh/
7z -p$password x $spath/tmp-ssh-files/dot_ssh.7z
popd

rm -rf $spath/tmp-ssh-files/

echo -n "Do you want to SKIP installing YCM for completion? Takes several mins. (Y/n): "
read answer
if [[ "$answer" == [Yy]* ]]; then
    YCM="--skip-ycm"
else
    YCM=""
fi


mkdir -p $HOME/repo/
if [ -d $HOME/repo/joel-snips ]; then
	sudo rm -rf $HOME/repo/joel-snips.bak
	mv $HOME/repo/joel-snips $HOME/repo/joel-snips.bak
fi

pushd $HOME/repo/
git clone git@github.com:joelagnel/joel-snips.git
pushd joel-snips
sudo ./rcfiles/setuprc "$YCM" < $password 

popd
popd
