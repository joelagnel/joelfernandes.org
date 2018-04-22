---
layout: post
title: "Ftrace events mechanism"
date: 2016-06-18 22:29:26 -0700
comments: true
categories: 
---
Ftrace events are a mechanism that allows different pieces of code in the kernel to 'broadcast' events of interest. Such as a scheduler context-switch `sched_switch` for example. In the scheduler core's `__schedule` function, you'll see something like: `trace_sched_switch(preempt, prev, next);`
This immediately results in a write to a per-cpu ring buffer storing info about what the previous task was, what the next one is, and whether the switch is happening as a result of kernel preemption (versus happening for other reasons such as a task waiting for I/O completion).

Under the hood, these ftrace events are actually implemented using tracepoints. The terms events are tracepoints appear to be used interchangeably, but it appears one could use a trace point if desired without having to do anything with ftrace. Events on the other hand use ftrace.

Let's discuss a bit about how a tracepoint works. Tracepoints are hooks that are inserted into points of code of interest and call a certain function of your choice (also known as a function probe). Inorder for the tracepoint to do anything, you have to register a function using `tracepoint_probe_register`. Multiple functions can be registered in a single hook. Once your tracepoint is hit, all functions registered to the tracepoint are executed. Also note that if no function is registered to the tracepoint, then the tracepoint is essentially a NOP with zero-overhead. Actually that's a lie, there is a branch (and some space) overhead only although negligible.

Here is the heart of the code that executes when a tracepoint is hit:
```
#define __DECLARE_TRACE(name, proto, args, cond, data_proto, data_args) \
        extern struct tracepoint __tracepoint_##name;                   \
        static inline void trace_##name(proto)                          \
        {                                                               \
                if (static_key_false(&__tracepoint_##name.key))         \
                        __DO_TRACE(&__tracepoint_##name,                \
                                TP_PROTO(data_proto),                   \
                                TP_ARGS(data_args),                     \
                                TP_CONDITION(cond),,);                  \
                if (IS_ENABLED(CONFIG_LOCKDEP) && (cond)) {             \
                        rcu_read_lock_sched_notrace();                  \
                        rcu_dereference_sched(__tracepoint_##name.funcs);\
                        rcu_read_unlock_sched_notrace();                \
                }                                                       \
        }                                                   
```
The `static_key_false` in the above code will evaluate to false if there's no probe registered to the tracepoint.

Digging further, `__DO_TRACE` does the following in `include/linux/tracepoint.h`
```
#define __DO_TRACE(tp, proto, args, cond, prercu, postrcu)              \
        do {                                                            \
                struct tracepoint_func *it_func_ptr;                    \
                void *it_func;                                          \
                void *__data;                                           \
                                                                        \
                if (!(cond))                                            \
                        return;                                         \
                prercu;                                                 \
                rcu_read_lock_sched_notrace();                          \
                it_func_ptr = rcu_dereference_sched((tp)->funcs);       \
                if (it_func_ptr) {                                      \
                        do {                                            \
                                it_func = (it_func_ptr)->func;          \
                                __data = (it_func_ptr)->data;           \
                                ((void(*)(proto))(it_func))(args);      \
                        } while ((++it_func_ptr)->func);                \
                }                                                       \
                rcu_read_unlock_sched_notrace();                        \
                postrcu;                                                \
        } while (0)
```
There's a lot going on there, but main part is the loop that goes through all function pointers (probes) that were registered to the tracepoint and calls them one after the other.

Now, here's some secrets. Since all ftrace events are tracepoints under the hood, you can piggy back onto interesting events in your kernel with your own probes. This allows you to write interesting tracers. Infact this is precisely how blktrace works, and also is how SystemTap hooks into ftrace events.
[Checkout a module I wrote](https://github.com/joelagnel/joel-snips/blob/master/k-patches/cpuhists.diff) that hooks onto `sched_switch` to build some histograms. The code there is still buggy but if you mess with it and improve it please share your work.

Now that we know a good amount about tracepoints, ftrace events are easy.

An ftrace event being based on tracepoints, makes full use of it but it has to do more. Ofcourse, it has to write events out to the ring buffer.
When you enable an ftrace event using debug-fs, at that instant the ftrace events framework registers an "event probe" function at the tracepoint that represents the event. How? Using `tracepoint_probe_register` just as we discussed.

The code for this is in the file `kernel/trace/trace_events.c` in function `trace_event_reg`.

```
int trace_event_reg(struct trace_event_call *call,
                    enum trace_reg type, void *data)
{
        struct trace_event_file *file = data;

        WARN_ON(!(call->flags & TRACE_EVENT_FL_TRACEPOINT));
        switch (type) {
        case TRACE_REG_REGISTER:
                return tracepoint_probe_register(call->tp,
                                                 call->class->probe,
                                                 file);
...
```
The probe function `call->class->probe` for trace events is defined in the file `include/trace/trace_events.h` and does the job of writing to the ring buffer. In a nutshell, the code gets a handle into the ring buffer, does assignment of the values to the entry structure and writes it out. There is some magic going on here to accomodate arbitrary number of arguments but I am yet to figure that out.
```
static notrace void                                                     \
trace_event_raw_event_##call(void *__data, proto)                       \
{                                                                       \
        struct trace_event_file *trace_file = __data;                   \
        struct trace_event_data_offsets_##call __maybe_unused __data_offsets;\
        struct trace_event_buffer fbuffer;                              \
        struct trace_event_raw_##call *entry;                           \
        int __data_size;                                                \
                                                                        \
        if (trace_trigger_soft_disabled(trace_file))                    \
                return;                                                 \
                                                                        \
        __data_size = trace_event_get_offsets_##call(&__data_offsets, args); \
                                                                        \
        entry = trace_event_buffer_reserve(&fbuffer, trace_file,        \
                                 sizeof(*entry) + __data_size);         \
                                                                        \
        if (!entry)                                                     \
                return;                                                 \
                                                                        \
        tstruct                                                         \
                                                                        \
        { assign; }                                                     \
                                                                        \
        trace_event_buffer_commit(&fbuffer);                            \
}
```
Let me know any comments you have or any other ftrace event behavior you'd like explained.
