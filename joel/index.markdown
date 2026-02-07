---
layout: page
title: false
comments: false
sharing: false
sidebar: false
footer: true
---
Hi! I'm Joel and I run this site! I'm a Linux kernel developer and work at NVIDIA. My interests are GPU software, Linux device drivers, scheduler, RCU, tracing, synchronization and kernel internals.

Follow me on [Twitter](https://twitter.com/joel_linux) and [LinkedIn](https://www.linkedin.com/in/joelagnel). Email me at: [joel@joelfernandes.org](mailto:joel@joelfernandes.org)


[Here's a list](https://patchwork.kernel.org/project/LKML/list/?submitter=170577) of recent kernel patches I submitted. I got [featured on hackaday](http://hackaday.com/2014/06/08/the-in-circuit-sd-card-switch/) and [have written for LWN](https://lwn.net/Articles/744522/) as well. [Check out my resume](/joel/joel-resume.pdf) and also see a list of past [talks and presentations](/resources).


Here's a list the full list of all articles I ever wrote:{% for post in site.posts %}
 <li><span>{{ post.date | date_to_string }}</span> &nbsp; <a href="{{ post.url }}">{{ post.title }}</a> [{{ post.categories | category_links }}] </li>
{% endfor %}
