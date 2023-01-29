---
layout: post
published: false
title: "androdeb: Running Powerful Linux tracing tools like BCC in Android"
comments: true
category: [linux, android, debian, tracing]
---
As a part of my work in the Android kernel team, I have been attempting to get
eBPF based BCC tracing tools working for our kernel developers. Recently we
developed and open sourced an eBPF daemon called BPFd and wrote about it on
LWN. A few users having been using it since. In this article we present another
solution called androdeb, which has many more advantages to BPFd: feature
completeness and easy of setup to name a few, while also having a few drawbacks
compared to BPFd. We go over an example androdeb workflow, talk about its
design, and discuss its advantages over BPFd, as well its drawbacks.

Android is the most popular Linux kernel based OS in the world. As successful
as it is, its userspace is more tailored towards running the Android framework
and Android apps efficiently, than being useful for a kernel or systems
developer to compile, develop and run kernel debugging tools.

The existing environment is also not suitable for running BCC/eBPF based tools
due to the fact the dependencies such as LLVM, clang and Python are not
available on device, as they are not needed for the operation of Android
userspace itself. Further, due to the recent security issues, there isn't much
of an incentive for security teams to support these tools on-device natively.

While I was developing BPFd, I also indepdently developed androdeb which tries
to solve the same problems BPFd does. It turned out that this technique is much
more powerful than BPFd and is now the preferred way of running BCC/eBPF tools
on Android that we recommend. It takes advantage of Android's design and layout
to easily run a custom userspace in it. This gives the systems engineer an
experience of developing and debugging similar to that of a regular Linux
desktop or server system. Further, no stoppage of the Android system is needed
inorder for this to work.

`androdeb` prepares an Android device with a debian based filesystem,
customizes and configures it according to the user's needs. Once the device is
prepared, `androdeb` starts a shell interface using the ADB protocol and gets
the user into a Linux shell environment that looks and feels like a typical
Linux server shell.

Once an Android device is connected to a host PC, the user clones androdeb
project and runs androdeb's prepare stage on their host like so:
```
androdeb prepare --download --bcc <path-to-kernel-source>
```
This is the quickest way to install androdeb, the typical output looks like:
```
Preparing device...

Downloading Androdeb from the web...

  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   606    0   606    0     0   1911      0 --:--:-- --:--:-- --:--:--  1905
100  293M  100  293M    0     0  11.9M      0  0:00:24  0:00:24 --:--:-- 15.1M
Using archive at /tmp/tmp.UrWBTv4cCq/androdeb-fs.tgz for filesystem preparation
Unpack of rootfs successful!
All done! Run "androdeb shell" to enter environment
```

This does a couple of things. It downloads a rootfs from `androdeb` project's
github page. Extracts the rootfs onto the device, and copies several setup
scripts into it. It then extracts kernel headers required for BCC to run from
`<path-to-kernel-sources>` and copies them over to the device. `androdeb` then
sets up environment variables to help BCC find the kernel headers on device. At
this point the preparion of the device is completed.

If one desires to build their custom own filesystem tailored for their needs,
They can :
```
androdeb prepare --editors --compilers --bcc
```
Notice that the `--download` parameter is skipped. This causes androdeb to
prepare a custom rootfs based on the configured options instead of pulling one
from github. For BCC, instead of packages I am cloning BCC master and building
it from source.

Once the device is prepared, the user runs `androdeb shell` to spawn a shell.

From the shell, The user can now see the list BCC tools available by running:
```
ls -l /usr/share/bcc/tools/
```

How it all works
================
Lets go through the all the components that make it work.

chroot
------
The `chroot(2)` system call runs a user specified command with a custom root
directory instead of `/`. This lets us run a root filesystem in Android made up
of a completley different userspace. androdeb's userspace is stored at
`/data/androdeb/debian/`. In this path, you'll see the typical directories
found in a regular Linux distribution such as `usr`, `lib`.

The `chroot(2)` implementation in the Linux kernel is quite simple yet clever.
Linux simply manages the path look up slightly different in the VFS code to
account for the new root directory. Luckily the chroot command ships with
Android by default on all recent Android systems.

debootstrap
-----------
Debian's `debootstrap` command is what `androdeb` uses to build a custom root
filesystem for a specific architecture.

`debootstrap` starts by downloading and unpackage packages to build the rootfs,
as the first stage. However, to run the second stage of the build, the packages
have to be configured while running in the target architecture's environment.
A wrapper `qemu-debootstrap` is available which cleverly manages this with the
help of Qemu's user binary emulation support. `debootstrap` copies over a
target architecture version of the `debootstrap` command itself, and executes
it using Qemu user emulation. Infact it runs the emulated binary using `chroot`
itself so that the second stage scripts runs with the rootfs being prepared as
the root, instead of the host's rootfs!

Android's directory structure
-----------------------------
Luckily in Android, the framework binaries such as `surfaceflinger` for
graphics or `system_server` for system services, reside in the `/system`
directory partition.

Inorder for debugging tools running in the chrooted environment to be able to
analyze symbol information from these binaries, it is important that the
`/system/` partition to also appears in the chrooted environment. To make this
happen, the androdeb startup script does a bind mound of Android's `/system`
partition to the chrooted path's `/system` partition. Other partitions such as
`/data/` and `/vendor/` are also similarly mounted. It would have possibly been
much trickier to get things to work if Android's framework binaries were in
`/usr` but thankfully that's not the case.

Conclusion
----------
A great advantage of running BCC within `androdeb` instead of running BCC with
`BPFd`, is that can run with any front end, not just BCC. This enables tools such
as `bpftrace` to work. Running BCC this way also means no tools to support a
host/target boundary split like BPFd does, and everything just works naturally.

A drawback would be quite a lot of free space (200-500MB) is needed in the
`/data/` partition for `androdeb` to work. This however is not an issue for
current generation Pixel devices. For devices with much lesser free space, BPFd
could be used in instead.
