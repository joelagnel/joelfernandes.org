---
title: false
comments: false
sharing: false
sidebar: false
footer: true
type: page
---
Hi! I'm Joel and I run this site! I'm a Linux kernel developer and work at Google. My interests are scheduler, tracing, synchronization and kernel internals.

Follow me on [Twitter](https://twitter.com/joel_linux), [Google+](https://plus.google.com/102415785508850230338) and [LinkedIn](https://www.linkedin.com/in/joelagnel). Email me at: [joel@linuxinternals.org](mailto:joel@linuxinternals.org)


[Here's a list](https://patchwork.kernel.org/project/LKML/list/?submitter=170577) of recent kernel patches I submitted. I got [featured on hackaday](http://hackaday.com/2014/06/08/the-in-circuit-sd-card-switch/) and [have written for LWN](https://lwn.net/Articles/744522/) as well. [Check out my resume](/joel/joel-resume.pdf) and also see a list of past [talks and presentations](/resources).


**[LinuxInternals.org](/linuxinternals/)** is a resource I created as a collection of articles and resources exploring Linux kernel and internals topics.

Here's a list the full list of all articles I ever wrote:
{{ range site.RegularPages }}
 <li><span>{{ .Date.Format "02 Jan 2006" }}</span> &nbsp; <a href="{{ .Permalink }}">{{ .Title }}</a> 
 {{ with .Params.categories }}
 [{{ if reflect.IsSlice . }}{{ range $index, $category := . }}<a href="{{ "/categories/" | relURL }}#{{ $category | urlize }}">{{ $category }}</a>{{ if ne $index (sub (len .) 1) }}, {{ end }}{{ end }}{{ else }}{{ $categories := split . ", " }}{{ range $index, $category := $categories }}<a href="{{ "/categories/" | relURL }}#{{ $category | urlize }}">{{ $category }}</a>{{ if ne $index (sub (len $categories) 1) }}, {{ end }}{{ end }}{{ end }}]
 {{ end }}
 </li>
{{ end }}