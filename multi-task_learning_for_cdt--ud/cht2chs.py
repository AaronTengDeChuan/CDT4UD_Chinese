from langconv import *
import codecs
import sys

print (sys.version)
print (sys.version_info)

def cht2chs(line):
	line = Converter('zh-hans').convert(line)
	line.encode('utf-8')
	return line

if __name__ == "__main__":
	assert len(sys.argv) == 3, "python cht2chs.py input_file output_file"
	input_file = sys.argv[1]
	output_file = sys.argv[2]

	with codecs.open(input_file, "r", encoding="utf-8") as fi:
		lines = fi.read()
		with codecs.open(output_file, "w", encoding="utf-8") as fo:
			for line in lines:
				fo.write(cht2chs(line))
