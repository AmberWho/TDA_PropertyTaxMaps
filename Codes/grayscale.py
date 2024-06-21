"""
grayscale.py

Author:   Amber, Anbo Wu
Date:     June 2024
Project:  Topological Data Analysis in an Economic Context: Property Tax Maps

Command:  python grayscale.py path (newpath)
          path - （file path and） name of original image, excluding filename extension
          newpath - optional, （file path and） name of output grayscale image, excluding filename extention
          
Description:
  Compatible with two sets of color legend offered by PropertyShark
  and transform sample images to a clean uniform-grayscale version.
"""

from PIL import Image
import sys

# open up a file, ready for processing.
if (len(sys.argv) == 2): 
  file = sys.argv[1]
  file2 = sys.argv[1]
elif (len(sys.argv) == 3):
  file = sys.argv[1]
  file2 = sys.argv[2]
else:
  sys.exit('Invalid arguments. Try [python grayscale.py path] or [python grayscale.py path newpath], excluding filename extension.')
  
# select legend type based on city index.
city = int(file[:2])
match city:
    case 4 | 5 | 8 | 9 | 11 | 12 | 13 | 14 | 17 | 20 | 21:
        _type = 'pink'
    case 1 | 2 | 3 | 6 | 7 | 10 | 15 | 16 | 18 | 19:
        _type = 'orange'
    case _:
        sys.exit('Invalid city index. Valid range: 1-21.')

im = Image.open(file + '.png')

def legend (num):
  # returns a color corresponding to the specified rank and legend type.
  # num = integer within range [0,11]
  #   0 means the lowest tax level, 11 means the highest
  # _type = {'pink','orange'}
  #   'pink' - red~pink~darkblue legend
  #   'orange' - red~orange~grayblue legend
  if (_type == 'pink'):
    legends = [(242,242,242),
                (0,38,83),
                (0,81,186),
                (0,142,214),
                (96,175,221),
                (191,209,229),
                (249,191,193),
                (252,140,153),
                (252,94,114),
                (206,17,38),
                (172,31,44),
                (124,33,40)]
  if (_type == 'orange'):
    legends = [(242,242,242),
                (49,77,112),
                (108,130,150),
                (142,162,181),
                (181,206,212),
                (190,210,187),
                (239,239,190),
                (223,209,178),
                (250,184,132),
                (237,117,82),
                (163,61,61),
                (120,42,42)]
  return legends[num]
    
def layer (im, num, c, newim):
  # read original graph (allow error in detected colors).
  # if not call fill() before return, returns the original graph.
  im_rgb = im.convert("RGB")
  b = 7
  newimg = Image.new('RGB', im.size, color = 'black')
  whitecolor = (255,255,255)
  for t in range (0,num+1,1):
    if t == 0:
      b = 2 # change this parameter for different error range
    target = legend(t)
    for x in range (0,im.size[0],1):
      for y in range (0,im.size[1],1):
        pixel = im_rgb.getpixel((x,y))
        if (target[0]-b <= pixel[0] <= target[0]+b
          and target[1]-b <= pixel[1] <= target[1]+b
          and target[2]-b <= pixel[2] <= target[2]+b):
          newimg.putpixel((x,y), whitecolor)
  newim = fill(newimg, newim, c)
  return newim

def fill (img, new, c):
  # get a black-and-white picture and omit noises:
  # check a 6x6 area centered with this point, if the white
  # pixels exceed number of 28 (can be changed), then this
  # should be a white point in the result graph
  # if not call widen() before return, returns a cleaned thin graph.
  img_rgb = img.convert("RGB")
  newim = Image.new('RGB', im.size, color = 'black')
  blackcolor = (0,0,0)
  whitecolor = (255,255,255)
  for x in range (0,im.size[0],1):
    for y in range (0,im.size[1],1):
      pixel = img_rgb.getpixel((x,y))
      if pixel == whitecolor:
        count = 0
        a = x-2
        b = y-2
        for t in range(6):
          for s in range(6):
            if (0 <= a+t < im.size[0] and 0 <= b+s < im.size[1]):
              check = img_rgb.getpixel((a+t,b+s))
              if check == whitecolor:
                count = count+1
        if count < 28:
          newim.putpixel((x,y), blackcolor)
        else:
          newim.putpixel((x,y), whitecolor)
  new = widen(newim, new, c)
  return new

def widen (img, newim, c):
  # widen the white area by certain range (can be modified)
  # so that walls between properties can be taken away.
  img_rgb = img.convert("RGB")
  whitecolor = (255,255,255)
  for x in range (0,im.size[0],1):
    for y in range (0,im.size[1],1):
      pixel = img_rgb.getpixel((x,y))
      if pixel == whitecolor:
        for t in range(-5,6,1):
          for s in range(-5,6,1):
            if (0 <= x+t < im.size[0] and 0 <= y+s < im.size[1]):
              newim.putpixel((x+t,y+s), c)
  return newim

# grayscale legend, sequenced from highest to lowest
c = [(255,255,255),
     (235,235,235),
     (215,215,215),
     (195,195,195),
     (175,175,175),
     (155,155,155),
     (135,135,135),
     (115,115,115),
     (95,95,95),
     (75,75,75),
     (55,55,55),
     (35,35,35),
     (25,25,25)]

newim = Image.new('RGB', im.size, color = 'black')
for a in range(11,-1,-1):
    newim = layer(im, a, c[12-a], newim)
    if a == 0:
        print("|=  Finish! (,,•ω•,,)  -=-------")
    else:
        print("==" + str('%4d' % (((12-a)/12)*100)) +"% ╰(●’◡’●)╮  ==---=-------")
newim.save(file2 + "gray.png")