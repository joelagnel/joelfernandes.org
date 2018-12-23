---
layout: post
title: "Dumping User and Kernel stacks on Kernel events"
date: 2014-04-24 20:28:24 -0500
comments: true
categories: linuxinternals
---

# Dumping User and Kernel stacks on Kernel events

Dumping the native kernel and userspace stack when a certain path in the kernel
or userspace occurs, can be useful to understand which code paths triggered a
certain behavior that you're trying to debug, such as an error you found in the
log. One such case is when you notice Selinux denial messages in logs but want
to know which path triggered it.

In this article we will show you how to use kernel instrumentation and BCC to
dump the both the user and kernel stack. The article applies both to Android
and regular Linux kernels.

## Example: Understanding which path triggered an SELinux denial

### Step 1: Add a tracepoint to the kernel
Apply the following diff to your kernel. It adds a tracepoint at precisely the
point where an SELinux denial is logged. If not cleanly applying, patch it in
manually.

```
diff --git a/include/trace/events/selinux.h b/include/trace/events/selinux.h
new file mode 100644
index 000000000000..dac185062634
--- /dev/null
+++ b/include/trace/events/selinux.h
@@ -0,0 +1,34 @@
+#undef TRACE_SYSTEM
+#define TRACE_SYSTEM selinux
+
+#if !defined(_TRACE_SELINUX_H) || defined(TRACE_HEADER_MULTI_READ)
+#define _TRACE_SELINUX_H
+
+#include <linux/ktime.h>
+#include <linux/tracepoint.h>
+
+TRACE_EVENT(selinux_denied,
+
+	TP_PROTO(int cls, int av),
+
+	TP_ARGS(cls, av),
+
+	TP_STRUCT__entry(
+		__field(	int,		cls	)
+		__field(	int,		av	)
+	),
+
+	TP_fast_assign(
+		__entry->cls = cls;
+		__entry->av = av;
+	),
+
+	TP_printk("denied %d %d",
+		__entry->cls,
+		__entry->av)
+);
+
+#endif /* _TRACE_SELINUX_H */
+
+/* This part ust be outside protection */
+#include <trace/define_trace.h>
diff --git a/security/selinux/avc.c b/security/selinux/avc.c
index 84d9a2e2bbaf..ab04b7c2dd01 100644
--- a/security/selinux/avc.c
+++ b/security/selinux/avc.c
@@ -34,6 +34,9 @@
 #include "avc_ss.h"
 #include "classmap.h"
 
+#define CREATE_TRACE_POINTS
+#include <trace/events/selinux.h>
+
 #define AVC_CACHE_SLOTS			512
 #define AVC_DEF_CACHE_THRESHOLD		512
 #define AVC_CACHE_RECLAIM		16
@@ -713,6 +716,12 @@ static void avc_audit_pre_callback(struct audit_buffer *ab, void *a)
 	struct common_audit_data *ad = a;
 	audit_log_format(ab, "avc:  %s ",
 			 ad->selinux_audit_data->denied ? "denied" : "granted");
+
+	if (ad->selinux_audit_data->denied) {
+		trace_selinux_denied(ad->selinux_audit_data->tclass,
+				     ad->selinux_audit_data->audited);
+	}
+
 	avc_dump_av(ab, ad->selinux_audit_data->tclass,
 			ad->selinux_audit_data->audited);
 	audit_log_format(ab, " for ");
```

### Step 2: Install [adeb](https://github.com/joelagnel/adeb)
Run the command:
```
adeb prepare --full
```
This also installs BCC on the Android device which contains the 'trace' utility
we need for the next step. For regular Linux kernels, you may have to [manually
install BCC](https://github.com/joelagnel/bcc/blob/master/README.md) or find a
package for it.

### Step 3: Start tracing the user and kernel stacks
Running the following command:
```
adeb shell
trace -K -U 't:selinux:selinux_denial'
```

You should see something like this when denials are triggered:
```
2286    2434    Binder:2286_4   selinux_denied   

        avc_audit_pre_callback+0xd8 [kernel]
        avc_audit_pre_callback+0xd8 [kernel]
        common_lsm_audit+0x64 [kernel]
        slow_avc_audit+0x74 [kernel]
        avc_has_perm+0xb8 [kernel]
        selinux_binder_transfer_file+0x158 [kernel]
        security_binder_transfer_file+0x50 [kernel]
        binder_translate_fd+0xcc [kernel]
        binder_transaction+0x1b64 [kernel]
        binder_ioctl+0xadc [kernel]
        do_vfs_ioctl+0x5c8 [kernel]
        sys_ioctl+0x88 [kernel]
        __sys_trace_return+0x0 [kernel]
        __ioctl+0x8 [libc.so]
        android::IPCThreadState::talkWithDriver(bool)+0x104 [libbinder.so]
        android::IPCThreadState::waitForResponse(android::Parcel*, int*)+0x40
                                                            [libbinder.so]
        android::IPCThreadState::executeCommand(int)+0x460 [libbinder.so]
        android::IPCThreadState::getAndExecuteCommand()+0xa0 [libbinder.so]
        android::IPCThreadState::joinThreadPool(bool)+0x40 [libbinder.so]
        [unknown] [libbinder.so]
        android::Thread::_threadLoop(void*)+0x12c [libutils.so]
        android::AndroidRuntime::javaThreadShell(void*)+0x90 [libandroid_runtime.so]
        __pthread_start(void*)+0x28 [libc.so]
        __start_thread+0x48 [libc.so]
```

The same trick can be used for dumping the stack on syscalls, random kernel
functions using kprobes and more! Just change the arguments passed to the
'trace' command.
