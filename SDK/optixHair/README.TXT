
Keyboard commands:
* 'q' or 'ESC': Quit the application.
* '1' linear (segments) interpretation of the geometry.
* '2' quadratic b-spline interpretation of the geometry.
* '3' cubic b-spline interpretation of the geometry.
* 's' "segment u": modulate base color with segment u, i.e. each segment starts
      black and ends with full luminance of the base color.
* 'r' "root-to-tip u": modulate base color with root-to-tip u, i.e. start hair
      at root black and full luminance of base color at hair's tip.
* 'i' "index color": assign one of six solid colors (red, green, blue, magenta,
      cyan, and yellow) based on a hair's index. The shader in this mode
      demonstrates how to compute a hair index from the primitive index.
      It also does root to tip shading but uses index based math to compute a
      contiguous u along the hair.
* 'c' "constant radius" hair geometry.
* 't' "tapered radius" hair geometry.

The hair model file for this SDK sample are from Cem Yuksel's web page:

    www.cemyuksel.com/research/hairmodels

You can download three additional .hair definitions from there to be used
by the sample code. Other hair files than the default can be passed to
the executable via its "--hair" command-line parameter.
