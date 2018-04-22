---
layout: post
title: "MicroSD card remote switch"
date: 2014-06-04 06:12:55 -0500
comments: true
categories: 
---
Recently, I've been wanting to remotely be able to program a MicroSD card with a new bootloader or filesystem *without* removing the card from its embedded target board (such as a Beaglebone or Pandaboard). Due to the lack of any such existing tools, I decided to design my own board. Finally have got it working, below are some pictures and a screencast demo video of the switcher in action! I sprinkled some power and status LED to show the user what's going on.

The base board requires two [SparkFun MicroSD sniffers](https://www.sparkfun.com/products/9419). The card cage on the sniffer is unused for my purposes. The switcher is controlled through an [FTDI cable](https://www.sparkfun.com/products/9717). 
I also [wrote up](https://github.com/joelagnel/microsd-switch/blob/master/sw/switch.c) a `switch` program to control the switcher with libftdi. You just have to pass to it the FTDI's serial number and whether you want to switch to host-mode (for programming the card) or target-mode (for booting the programmed card).
[Hardware design files](https://github.com/joelagnel/microsd-switch) are available under a CC-BY-NC-SA 3.0 license.

Screencast{% youtube StpIihVQ7oM %}

Pictures
{% img https://raw.githubusercontent.com/joelagnel/microsd-switch/master/board-pics/microsd-inaction/photo5.jpg %}
{% img https://raw.githubusercontent.com/joelagnel/microsd-switch/master/board-pics/microsd-inaction/photo2.jpg %}
{% img https://raw.githubusercontent.com/joelagnel/microsd-switch/master/board-pics/front.png %}

Hope you enjoyed it, let me know what yout think in the comments:)
