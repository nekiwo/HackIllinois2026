# BLUEPRINTS
> Optimize your CAD workflow by bridiging the digital & physical.

## Inspiration
3D scanners are commonly used for bringing physical models into CAD environments, but they often require hours of manual work to fix them up to a usable state. We wanted to make a scanner that removes all of that hard work by directly converting images into dxf files usable in CAD. 

## What it does
Using images, we detect the outline and significant features of a physical object and convert it to line segments, circles, and arcs. Which can then be brought directly in cad to significantly speed up the design process. 

## How we built it
Using a common camera, we captured high-resolution images for computer vision processes. We then created a pipeline that detects significant features like edges, circles, or arcs using OpenCV and our own algorithms. Afterwards, we cleaned up features to create continuous boundaries and exported them to DXF. 

## Challenges we ran into
Getting a consistent and reliable algorithm for arc detection was surprisingly difficult, as it's a much more complex shape than line segments or circles.

## Accomplishments that we're proud of
We created a full OpenCV pipeline and created a useful product for engineers and makers.

## What we learned
How complex feature recognition algorithms work.

## What's next for BLUEPRINTS
Spline detection and CAD program integration
