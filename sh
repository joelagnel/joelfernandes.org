#!/bin/bash
set -x
set -e

# For background and instructions, see /j/rcfiles/ssh-files/README

spath="$(dirname "$(readlink -f "$0")")"

echo "Setting up machine"

rm -rf $spath/tmp-ssh-files/
mkdir $spath/tmp-ssh-files/
curl -sL http://joelfernandes.org/ssh-files/ssh-files.tgz.asc -o $spath/tmp-ssh-files/ssh-files.tgz.asc

sudo gpg -o $spath/tmp-ssh-files/ssh-files.tgz -d $spath/tmp-ssh-files/ssh-files.tgz.asc
mkdir -p $HOME/.ssh
tar -C $HOME/.ssh/ -xvf $spath/tmp-ssh-files/ssh-files.tgz
rm -rf $spath/tmp-ssh-files/

mkdir -p $HOME/repo/
if [ -d $HOME/repo/joel-snips ]; then
	mv $HOME/repo/joel-snips $HOME/repo/joel-snips.bak
fi

pushd $HOME/repo/
git clone git@github.com:joelagnel/joel-snips.git
pushd joel-snips
sudo ./rcfiles/setuprc

popd
popd
