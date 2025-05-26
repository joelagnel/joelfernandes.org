---
layout: post
date: 
title: "SELinux Debugging on ChromeOS"
comments: true
categories: [selinux, debugging]
---
Not being an SELinux expert but having to deal with it from time-to-time, I find it may be productive to write some debugging notes. So this post is about an issue that I fixed just this morning.

## Issue: Runtime labeled pseudo-files are not getting labelled

On ChromeOS, there are Android-specific policies which are combined with the ChromeOS ones. For someone who tries to stay away from SELinux, this is pretty much voodoo. 

Let me start with the last issue. I had to label a procfs file added by a test kernel patch which I am developed. This procfs would then be toggled by ChromeOS’s experiment system to enable the feature and collect data. The procfs file was called `/proc/sys/kernel/timer_highres`.

ChromeOS’s SELinux policies live in a repository at `src/platform2/sepolicy` as of this writing. In this repo, there is a file named`genfs_contexts` which contains `genfscon` rules.

`genfscon` is used to dynamically generate security contexts for pseudo-filesystems such as `procfs`. This is documented more in the [docs here](https://selinuxproject.org/page/FileStatements#File_System_Labeling_Statements).

The main issue here was I was making changes to the wrong `genfs_contexts` file to begin with. That’s easy to do when there are 2 of them. But here’s how I found out that my changes were not getting through to the device:

1. Make [the change](https://chromium-review.googlesource.com/c/chromiumos/platform2/+/4568967/1/sepolicy/policy/base/genfs_contexts) to `genfs_contexts` file to label the proc node I am interested in.
2. After building and installing the `platform-base/selinux-policy` ChromeOS package, inspect the device.
3. Where to inspect? Read the [eBuild file sources](https://source.chromium.org/chromiumos/chromiumos/codesearch/+/main:src/third_party/chromiumos-overlay/chromeos-base/selinux-policy/selinux-policy-9999.ebuild;l=312?q=selinux-policy%20ebuild) which is what builds that package. From this I figured that the `genfs_contexts` was being compiled into a binary located at `/etc/selinux/arc/policy.30`. I am still not sure what the numeric suffix means, but it does not matter.
4. Next, use `grep` to scan this binary for `timer_highres`, I found that the `grep` brought back nothing. Confirming that my change had no effect.
5. Hmm, what if the string was actually some weird encoding and `grep` could not find it? After all, I know nothing about that binary file’s format.
    1. So copy policy binary file and confirm with `seinfo` command on a regular Linux machine: 
    2. First `scp` the `policy.30` file from the device.
    3. Then on regular Linux, run: `seinfo --genfscon=proc policy.30 | less`.
    4. Look for the `genfscon` line for `timer_highres` in the output. 
6. [Modify](https://chromium-review.googlesource.com/c/chromiumos/platform2/+/4606196) the correct `genfs_contexts` file and repeat.
7. Use `ls -lZ` to confirm that the procfs file is now correctly labeled.

## Conclusions and Lessons Learnt

1. Trial and error approach to debugging is OK, but always make sure your changes are being reflected on what is on the device, otherwise you’ll waste a lot of time.
2. Once a theory is validated, example: “`genfscon` rule did not get updated”, then stop with step #1, and focus on digging deeper into the validated theory which we know is de-facto correct.
3. Read the build source files to get a better understanding of how your code changes result in the artifacts. That’s how I learnt the `genfscon` rule I was adding was getting compiled into the `policy.30` file.
4. Document the issue like in this blog post, and also in the source file being changed so that others don’t run into the issue in the future.
