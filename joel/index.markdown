---
layout: page
title: false
comments: false
sharing: false
sidebar: false
footer: true
---
Hi! I'm Joel and I run this site! I'm a Linux kernel developer and work in the Android kernel team at Google. My interests are scheduler, tracing, synchronization and kernel internals. I am a permanent Linux kernel contributor.

Follow me on [Twitter](https://twitter.com/joel_linux), [Google+](https://plus.google.com/102415785508850230338) and [LinkedIn](https://www.linkedin.com/in/joelagnel). Email me at: [joel@linuxinternals.org](mailto:joel@linuxinternals.org)


[Here's a list](https://patchwork.kernel.org/project/LKML/list/?submitter=170577) of recent kernel patches I submitted. I got [featured on hackaday](http://hackaday.com/2014/06/08/the-in-circuit-sd-card-switch/) and [have written for LWN](https://lwn.net/Articles/744522/) as well. [My resume](/joel/joel-resume.pdf) covers a lot about my background and work experience.


**[LinuxInternals.org](/linuxinternals/)** is a resource I created as a collection of articles and resources exploring Linux kernel and internals topics.

Here's a list the full list of all articles I ever wrote:{% for post in site.posts %}
 <li><span>{{ post.date | date_to_string }}</span> &nbsp; <a href="{{ post.url }}">{{ post.title }}</a> [{{ post.categories | category_links }}] </li>
{% endfor %}
