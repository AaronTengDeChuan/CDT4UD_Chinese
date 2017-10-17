#! /usr/bin/python

import sys

if len(sys.argv) < 1:
  print "usage: [training set]"
  exit()
fi = open(sys.argv[1],"r")
fi = fi.read()
fi = fi.strip().split("\n\n")
print len(fi),